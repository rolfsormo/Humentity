use crate::{
    loaders::{MhcloAsset, MhcloVertexMap},
    mesh_ops::{
        fix_normals, fix_normals_multiple, generate_mhid_lookup, generate_vertex_map,
        get_uv_coords, get_vertex_normals, get_vertex_positions, get_vertex_tangents,
        set_asset_rig_arrays,
    },
    morphs::adjust_helpers_to_morphs,
    prelude::*,
    rigs::RigSpec,
};
use ahash::AHashMap;
use bevy::mesh::morph::MorphAttributes;
use bevy::{asset::RenderAssetUsages, prelude::*};
use std::sync::{Arc, RwLock};

/// Overrides the template shapes for a part
#[derive(Component, Deref, Clone, Eq, PartialEq, Hash, Debug)]
pub struct TemplateOverride(pub Handle<CharacterTemplate>);

/// Collection of parts that should be stitched together.  This will
/// spawn siblings for each part then despawn this entity.
#[derive(Component, Clone, Deref, DerefMut, Eq, PartialEq, Hash, Debug)]
pub struct StitchedParts(pub Vec<StitchedPart>);

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct StitchedPart {
    pub(crate) part: Handle<MhcloAsset>,
    pub(crate) template_override: Option<TemplateOverride>,
    pub(crate) lod: usize,
}

impl From<Handle<MhcloAsset>> for StitchedPart {
    fn from(part: Handle<MhcloAsset>) -> Self {
        Self {
            part,
            template_override: None,
            lod: 0,
        }
    }
}

impl StitchedPart {
    pub fn with_template_override(mut self, template: Handle<CharacterTemplate>) -> Self {
        self.template_override = Some(TemplateOverride(template));
        self
    }

    pub const fn with_lod(mut self, lod: usize) -> Self {
        self.lod = lod;
        self
    }
}

pub fn shape_mesh_from_helpers_mhclo(
    mesh: &Mesh,
    mhclo: &MhcloAsset,
    helpers: &[Vec3],
    mhid_lookup: &[u16],
    bevy_vertex_map: &AHashMap<u16, Vec<u16>>,
) -> Mesh {
    shape_mesh_from_helpers_mhclo_scaled(
        mesh,
        mhclo,
        helpers,
        helpers,
        mhid_lookup,
        bevy_vertex_map,
    )
}

/// Fit a proxy to `helpers`, scaling its vertex offsets by the reference
/// distances measured on `scale_helpers`.
pub fn shape_mesh_from_helpers_mhclo_scaled(
    mesh: &Mesh,
    mhclo: &MhcloAsset,
    helpers: &[Vec3],
    scale_helpers: &[Vec3],
    mhid_lookup: &[u16],
    bevy_vertex_map: &AHashMap<u16, Vec<u16>>,
) -> Mesh {
    let offset_scale = mhclo.get_offset_scale_mhclo(scale_helpers);
    let mut vertices = get_vertex_positions(mesh);
    for (vert, mh_asset_vertex) in mhid_lookup.iter().enumerate() {
        let hm = &mhclo.helper_map[*mh_asset_vertex as usize];
        match hm {
            MhcloVertexMap::SingleVertex(i) => {
                vertices[vert] = helpers[*i as usize];
            }
            MhcloVertexMap::Triangle {
                helper_verts,
                helper_weights,
                helper_offset,
            } => {
                let mut position = Vec3::ZERO;
                for i in 0..3 {
                    let mh_vert = helper_verts[i];
                    let wt = helper_weights[i];
                    if let Some(mh_helper_position) = helpers.get(mh_vert as usize) {
                        position += *mh_helper_position * wt;
                    }
                }
                let offset = offset_scale * helper_offset;
                vertices[vert] = position + offset;
            }
        }
    }

    let mut mesh = Mesh::new(
        bevy::mesh::PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, get_uv_coords(mesh))
    .with_inserted_indices(mesh.indices().unwrap_or_else(|| {
        panic!("proxy mesh has no index buffer: humentity meshes must be indexed triangle lists (check the proxy .obj)")
    }).clone())
    .with_computed_area_weighted_normals();
    // Tangents come from the seam-averaged normals, so they are generated once,
    // after the averaging.
    fix_normals(&mut mesh, bevy_vertex_map)
        .expect("tangent generation failed on the morphed proxy mesh: degenerate triangles after morphing (check extreme morph weights)");
    mesh
}

/// Fit one template shape onto `base`, the proxy fitted to the base mesh.
///
/// A motion shape (`keep_proxy_scale`) adds only its motion: both its fit and
/// a reference fit to the base mesh use the first shape's offset scale, and
/// their difference is applied to `base`. The reference depends only on the
/// base mesh and that scale, so it is fitted once into `reference` and shared
/// by every motion shape of the template.
#[allow(clippy::too_many_arguments)] // Fitting inputs come from the build call sites unchanged.
fn fit_template_shape(
    base: &Mesh,
    mhclo: &MhcloAsset,
    shape: &CharacterMorphShape,
    helpers: &[Vec3],
    first_helpers: &[Vec3],
    basemesh: &[Vec3],
    reference: &mut Option<Mesh>,
    mhid_lookup: &[u16],
    bevy_vertex_map: &AHashMap<u16, Vec<u16>>,
) -> Mesh {
    if !shape.keep_proxy_scale {
        return shape_mesh_from_helpers_mhclo(base, mhclo, helpers, mhid_lookup, bevy_vertex_map);
    }
    let fit = |helpers: &[Vec3]| {
        shape_mesh_from_helpers_mhclo_scaled(
            base,
            mhclo,
            helpers,
            first_helpers,
            mhid_lookup,
            bevy_vertex_map,
        )
    };
    let reference = &*reference.get_or_insert_with(|| fit(basemesh));
    let moved = fit(helpers);
    let shift = |base: Vec<Vec3>, moved: Vec<Vec3>, reference: Vec<Vec3>| {
        base.into_iter()
            .zip(moved.into_iter().zip(reference))
            .map(|(base, (moved, reference))| base + moved - reference)
            .collect::<Vec<_>>()
    };
    let positions = shift(
        get_vertex_positions(base),
        get_vertex_positions(&moved),
        get_vertex_positions(reference),
    );
    let normals = shift(
        get_vertex_normals(base),
        get_vertex_normals(&moved),
        get_vertex_normals(reference),
    );
    let Some(bevy::mesh::VertexAttributeValues::Float32x4(base_tangents)) =
        base.attribute(Mesh::ATTRIBUTE_TANGENT)
    else {
        panic!("the base proxy mesh has no tangents: fitted proxies always generate them");
    };
    let tangent_shift = shift(
        base_tangents
            .iter()
            .map(|t| Vec3::new(t[0], t[1], t[2]))
            .collect(),
        get_vertex_tangents(&moved).expect("fitted proxies always have tangents"),
        get_vertex_tangents(reference).expect("fitted proxies always have tangents"),
    );
    let tangents = tangent_shift
        .into_iter()
        .zip(base_tangents)
        .map(|(t, base)| [t.x, t.y, t.z, base[3]])
        .collect::<Vec<_>>();
    let mut mesh = base.clone();
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, tangents);
    mesh
}

/// Fit `base`, the proxy already fitted to the base mesh, to every shape of
/// `template`, in order. `context` names the build in error messages.
#[allow(clippy::too_many_arguments)] // Fitting inputs come from the build call sites unchanged.
fn fit_template_shapes(
    base: &Mesh,
    mhclo: &MhcloAsset,
    template: &CharacterTemplate,
    mh_morphs: &Arc<RwLock<AHashMap<&'static str, TargetAsset>>>,
    basemesh: &[Vec3],
    mhid_lookup: &[u16],
    vertex_map: &AHashMap<u16, Vec<u16>>,
    context: &str,
) -> Vec<Mesh> {
    let mut first_helpers = None;
    let mut reference = None;
    template
        .shapes
        .iter()
        .map(|shape| {
            let helpers = adjust_helpers_to_morphs(&shape.morphs, mh_morphs, basemesh)
                .unwrap_or_else(|e| panic!("{context} shape '{}': {e} (check the shape's morph names against the morph targets and manifest)", shape.name));
            let first = first_helpers.get_or_insert_with(|| helpers.clone());
            fit_template_shape(
                base,
                mhclo,
                shape,
                &helpers,
                first,
                basemesh,
                &mut reference,
                mhid_lookup,
                vertex_map,
            )
        })
        .collect()
}

pub(crate) fn build_final_mesh_mhclo(
    mhclo: &MhcloAsset,
    input_mesh: &Mesh,
    mesh_verts: &ObjVertsAsset,
    template: &CharacterTemplate,
    mh_morphs: Arc<RwLock<AHashMap<&'static str, TargetAsset>>>,
    basemesh: Arc<Vec<Vec3>>,
    _rig_spec: &RigSpec,
    lod_bone_names: &[&'static str],
    lod_weights: &AHashMap<&'static str, AHashMap<u16, f32>>,
) -> (Mesh, Vec<String>) {
    let vertices = get_vertex_positions(input_mesh);
    let vertex_map = generate_vertex_map(&mesh_verts.vertices, &vertices);
    let mhid_lookup = generate_mhid_lookup(&vertex_map);

    let mut input_mesh =
        shape_mesh_from_helpers_mhclo(input_mesh, mhclo, &basemesh, &mhid_lookup, &vertex_map);

    let meshes = fit_template_shapes(
        &input_mesh,
        mhclo,
        template,
        &mh_morphs,
        &basemesh,
        &mhid_lookup,
        &vertex_map,
        "mesh build",
    );

    let mut morph_names = vec![];
    if !template.shapes.is_empty() {
        let base_positions = get_vertex_positions(&input_mesh);
        let base_normals = get_vertex_normals(&input_mesh);
        let base_tangents = get_vertex_tangents(&input_mesh).expect("Failed to get tangents");
        let mut morphs = vec![];

        for (is, shape_mesh) in meshes.iter().enumerate() {
            let mut morph = Vec::<MorphAttributes>::new();
            let shape_positions = get_vertex_positions(shape_mesh);
            let shape_normals = get_vertex_normals(shape_mesh);
            let shape_tangents =
                get_vertex_tangents(shape_mesh).expect("Shape meshes should always have tangents");

            for vtx in 0..base_positions.len() {
                morph.push(MorphAttributes::from([
                    shape_positions[vtx] - base_positions[vtx],
                    shape_normals[vtx] - base_normals[vtx],
                    shape_tangents[vtx] - base_tangents[vtx],
                ]));
            }

            morph_names.push(template.shapes[is].name.to_string());
            morphs.push(morph.into_iter());
        }
        let morph_attributes: Vec<MorphAttributes> = morphs.into_iter().flatten().collect();
        input_mesh.set_morph_targets(morph_attributes);
    }

    set_asset_rig_arrays(
        &mut input_mesh,
        &mhid_lookup,
        &mhclo.helper_map,
        lod_bone_names,
        lod_weights,
    );

    (input_mesh, morph_names)
}

pub(crate) fn build_final_meshes_mhclo(
    mhclos: &[MhcloAsset],
    input_meshes: &mut [Mesh],
    mesh_verts: &[ObjVertsAsset],
    templates: &[CharacterTemplate],
    mh_morphs: Arc<RwLock<AHashMap<&'static str, TargetAsset>>>,
    basemesh: Arc<Vec<Vec3>>,
    _rig_spec: &RigSpec,
    lod_bone_names: &[&'static str],
    lod_weights: &AHashMap<&'static str, AHashMap<u16, f32>>,
) -> (Vec<Mesh>, Vec<Vec<String>>) {
    let mut mhid_lookup = vec![];
    let mut vertex_map = vec![];

    let n_meshes = input_meshes.len();
    for i in 0..n_meshes {
        let mesh = &input_meshes[i];
        let vertices = get_vertex_positions(mesh);
        vertex_map.push(generate_vertex_map(&mesh_verts[i].vertices, &vertices));
        mhid_lookup.push(generate_mhid_lookup(&vertex_map[i]));
        input_meshes[i] = shape_mesh_from_helpers_mhclo(
            mesh,
            &mhclos[i],
            &basemesh,
            &mhid_lookup[i],
            &vertex_map[i],
        );
    }

    // Key by (mesh index, shape name): two templates may define the same shape
    // name with different morph maps, and each mesh needs its own fitted
    // variant. A bare shape-name key would share the first-built entry across
    // templates and meshes with different geometry.
    let mut shape_meshes: AHashMap<(usize, &'static str), Option<Mesh>> = AHashMap::default();
    for (i_mesh, mesh) in input_meshes.iter().enumerate() {
        let template = &templates[i_mesh];
        let mut mesh_shapes: Vec<Option<Mesh>> = fit_template_shapes(
            mesh,
            &mhclos[i_mesh],
            template,
            &mh_morphs,
            &basemesh,
            &mhid_lookup[i_mesh],
            &vertex_map[i_mesh],
            &format!("stitched mesh {i_mesh}"),
        )
        .into_iter()
        .map(Some)
        .collect();
        let mut tmp_mesh_vec = mesh_shapes
            .iter_mut()
            .filter_map(|m| m.as_mut())
            .collect::<Vec<_>>();
        fix_normals_multiple(&mut tmp_mesh_vec);
        for (mesh_shape, fitted) in template.shapes.iter().zip(mesh_shapes) {
            shape_meshes.insert((i_mesh, mesh_shape.name), fitted);
        }
    }

    let mut morph_names = vec![];

    for i_mesh in 0..n_meshes {
        let mut names = vec![];
        let template = &templates[i_mesh];

        if !template.shapes.is_empty() {
            let base_positions = get_vertex_positions(&input_meshes[i_mesh]);
            let base_normals = get_vertex_normals(&input_meshes[i_mesh]);
            let base_tangents =
                get_vertex_tangents(&input_meshes[i_mesh]).expect("Failed to get tangents");
            let mut morph_attrs = vec![];

            for shape in template.shapes.iter() {
                let Some(Some(shape_mesh)) = shape_meshes.get(&(i_mesh, shape.name)) else {
                    continue;
                };
                let mut morph = Vec::<MorphAttributes>::new();

                let shape_positions = get_vertex_positions(shape_mesh);
                let shape_normals = get_vertex_normals(shape_mesh);
                let shape_tangents = get_vertex_tangents(shape_mesh)
                    .expect("Shape meshes should always have tangents");

                for vtx in 0..base_positions.len() {
                    morph.push(MorphAttributes::from([
                        shape_positions[vtx] - base_positions[vtx],
                        shape_normals[vtx] - base_normals[vtx],
                        shape_tangents[vtx] - base_tangents[vtx],
                    ]));
                }

                names.push(shape.name.to_string());
                morph_attrs.push(morph.into_iter());
            }
            let morph_attributes: Vec<MorphAttributes> =
                morph_attrs.into_iter().flatten().collect();
            input_meshes[i_mesh].set_morph_targets(morph_attributes);
        }
        morph_names.push(names);

        set_asset_rig_arrays(
            &mut input_meshes[i_mesh],
            &mhid_lookup[i_mesh],
            &mhclos[i_mesh].helper_map,
            lod_bone_names,
            lod_weights,
        );
    }

    (input_meshes.to_vec(), morph_names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loaders::{MhcloAssetLoader, ObjVertsAsset, ObjVertsAssetLoader, ObjVertsSettings};
    use crate::morphs::MorphTargets;
    use bevy::asset::AssetPlugin;

    /// The proxies the characterisation covers: a body proxy (helpers only) and a
    /// garment (helper triangles with offsets).
    const PARTS: [&str; 2] = [
        "proxymeshes/proxy741/proxy741.proxy",
        "clothes/underwear/simple_briefs/simple_briefs.mhclo",
    ];

    struct Part {
        mhclo: MhcloAsset,
        mesh: Mesh,
        verts: ObjVertsAsset,
    }

    /// Load the base mesh helpers and [`PARTS`] from the crate's assets.
    fn load() -> (Vec<Vec3>, Vec<Part>) {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            AssetPlugin {
                file_path: format!("{}/assets", env!("CARGO_MANIFEST_DIR")),
                ..default()
            },
            bevy_obj::ObjPlugin,
        ))
        .init_asset::<Mesh>()
        .init_asset::<ObjVertsAsset>()
        .register_asset_loader(ObjVertsAssetLoader)
        .init_asset::<MhcloAsset>()
        .register_asset_loader(MhcloAssetLoader);
        let server = app.world().resource::<AssetServer>().clone();
        let basemesh: Handle<ObjVertsAsset> = server
            .load_builder()
            .with_settings(|settings: &mut ObjVertsSettings| settings.is_basemesh_helpers = true)
            .load("base.obj");
        let mhclos: Vec<Handle<MhcloAsset>> = PARTS.iter().map(|path| server.load(*path)).collect();
        let mut objs: Option<Vec<(Handle<Mesh>, Handle<ObjVertsAsset>)>> = None;
        for _ in 0..10_000 {
            app.update();
            let world = app.world();
            let mhclo_assets = world.resource::<Assets<MhcloAsset>>();
            if objs.is_none() && mhclos.iter().all(|h| mhclo_assets.contains(h)) {
                objs = Some(
                    mhclos
                        .iter()
                        .map(|h| {
                            let obj = mhclo_assets.get(h).expect("loaded").obj_file.clone();
                            (server.load(obj.clone()), server.load(obj))
                        })
                        .collect(),
                );
            }
            let meshes = world.resource::<Assets<Mesh>>();
            let verts = world.resource::<Assets<ObjVertsAsset>>();
            if let Some(objs) = &objs
                && verts.contains(&basemesh)
                && objs
                    .iter()
                    .all(|(mesh, v)| meshes.contains(mesh) && verts.contains(v))
            {
                let parts = mhclos
                    .iter()
                    .zip(objs)
                    .map(|(mhclo, (mesh, v))| Part {
                        mhclo: mhclo_assets.get(mhclo).expect("loaded").clone(),
                        mesh: meshes.get(mesh).expect("loaded").clone(),
                        verts: verts.get(v).expect("loaded").clone(),
                    })
                    .collect();
                return (
                    verts.get(&basemesh).expect("loaded").vertices.clone(),
                    parts,
                );
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        panic!("the test assets did not load");
    }

    /// Two deterministic targets touching every 7th and every 11th base vertex.
    fn targets(base_len: usize) -> Arc<RwLock<AHashMap<&'static str, TargetAsset>>> {
        let target = |name: &'static str, every: usize, offset: Vec3| TargetAsset {
            name,
            deltas: (0..base_len)
                .step_by(every)
                .map(|vertex| TargetDelta {
                    vertex: u16::try_from(vertex).expect("base mesh vertex fits u16"),
                    offset: offset * (1.0 + (vertex % 5) as f32 * 0.1),
                })
                .collect(),
        };
        let mut map = AHashMap::default();
        map.insert(
            "test-grow",
            target("test-grow", 7, Vec3::new(0.0, 0.02, 0.0)),
        );
        map.insert(
            "test-jaw",
            target("test-jaw", 11, Vec3::new(0.0, -0.01, 0.015)),
        );
        Arc::new(RwLock::new(map))
    }

    fn morphs(values: &[(&'static str, f32)]) -> MorphTargets {
        let mut morphs = MorphTargets::default();
        for &(name, value) in values {
            morphs.insert(name, value);
        }
        morphs
    }

    /// Fold every position, normal and tangent of `mesh` into `hash`, bit for bit.
    fn fold(hash: &mut u64, mesh: &Mesh) {
        let mut word = |w: u32| {
            *hash = (*hash ^ u64::from(w)).wrapping_mul(0x0100_0000_01b3);
        };
        for position in get_vertex_positions(mesh) {
            position.to_array().iter().for_each(|v| word(v.to_bits()));
        }
        for normal in get_vertex_normals(mesh) {
            normal.to_array().iter().for_each(|v| word(v.to_bits()));
        }
        let Some(bevy::mesh::VertexAttributeValues::Float32x4(tangents)) =
            mesh.attribute(Mesh::ATTRIBUTE_TANGENT)
        else {
            panic!("fitted meshes carry tangents");
        };
        for tangent in tangents {
            tangent.iter().for_each(|v| word(v.to_bits()));
        }
    }

    /// The fitted base and shape meshes of every test part, as one digest. Pins the
    /// fitting's output so that speeding it up provably changes nothing.
    #[test]
    fn fitted_shape_meshes_are_unchanged() {
        let (basemesh, parts) = load();
        let mh_morphs = targets(basemesh.len());
        let template = CharacterTemplate::new([
            CharacterMorphShape::new("body", morphs(&[("test-grow", 0.8)])),
            CharacterMorphShape::motion("jaw", morphs(&[("test-grow", 0.8), ("test-jaw", 1.0)])),
            CharacterMorphShape::motion("jaw-half", morphs(&[("test-jaw", 0.5)])),
        ]);
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for part in &parts {
            let vertices = get_vertex_positions(&part.mesh);
            let vertex_map = generate_vertex_map(&part.verts.vertices, &vertices);
            let mhid_lookup = generate_mhid_lookup(&vertex_map);
            let base = shape_mesh_from_helpers_mhclo(
                &part.mesh,
                &part.mhclo,
                &basemesh,
                &mhid_lookup,
                &vertex_map,
            );
            fold(&mut hash, &base);
            for shape in fit_template_shapes(
                &base,
                &part.mhclo,
                &template,
                &mh_morphs,
                &basemesh,
                &mhid_lookup,
                &vertex_map,
                "test",
            ) {
                fold(&mut hash, &shape);
            }
        }
        assert_eq!(
            hash, 0x7df7_60bf_9b36_eb38,
            "fitted shape digest {hash:#018x}"
        );
    }
}
