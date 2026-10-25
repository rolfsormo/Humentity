//! Morph-aware bounds for CPU-skinned character parts.
//!
//! Bevy's ordinary mesh bounds describe the undeformed vertices. Its dynamic
//! skin bounds handle joint motion but explicitly exclude morph targets. Build
//! the per-joint envelope from this instance's actual morphed positions, then
//! use Bevy's skin-bound transformation each frame. No render mesh duplication
//! or disabled frustum culling is needed.
use crate::spawn_mesh::CharacterPart;
use bevy::{
    asset::RenderAssetUsages,
    camera::{primitives::Aabb, visibility::VisibilitySystems},
    mesh::{
        VertexAttributeValues,
        morph::{MeshMorphWeights, MorphWeights},
        skinning::{
            SkinnedMesh, SkinnedMeshBoundsError, SkinnedMeshInverseBindposes,
            entity_aabb_from_skinned_mesh_bounds,
        },
    },
    prelude::*,
    render::render_resource::PrimitiveTopology,
    transform::TransformSystems,
};
use std::collections::HashSet;

#[derive(Component)]
struct PartBounds {
    source: AssetId<Mesh>,
    weights: Vec<f32>,
    // Only skinned_mesh_bounds is retained; there are no vertex buffers here.
    envelope: Mesh,
}

#[derive(Debug, thiserror::Error)]
enum BoundsError {
    #[error("character mesh is missing {0}")]
    Missing(&'static str),
    #[error("character morph data does not match its vertex/weight count")]
    MorphLayout,
    #[error("character morph positions or weights are non-finite")]
    NonFinite,
    #[error(transparent)]
    Skin(#[from] SkinnedMeshBoundsError),
}

pub(crate) fn install(app: &mut App) {
    app.add_systems(
        PostUpdate,
        update
            .after(TransformSystems::Propagate)
            .after(VisibilitySystems::CalculateBounds)
            .before(VisibilitySystems::CheckVisibility),
    );
}

fn prepare(id: AssetId<Mesh>, source: &Mesh, weights: &[f32]) -> Result<PartBounds, BoundsError> {
    let Some(VertexAttributeValues::Float32x3(base)) = source.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        return Err(BoundsError::Missing("POSITION"));
    };
    if base.is_empty() {
        return Err(BoundsError::MorphLayout);
    }
    if weights.iter().any(|weight| !weight.is_finite()) {
        return Err(BoundsError::NonFinite);
    }
    let mut positions = base.clone();
    if let Some(targets) = source.morph_targets() {
        if targets.len() != weights.len() * positions.len() {
            return Err(BoundsError::MorphLayout);
        }
        for (target, &weight) in targets.chunks_exact(positions.len()).zip(weights) {
            for (position, delta) in positions.iter_mut().zip(target) {
                *position = (Vec3::from_array(*position) + delta.position * weight).to_array();
            }
        }
    } else if !weights.is_empty() {
        return Err(BoundsError::MorphLayout);
    }
    if positions
        .iter()
        .any(|position| !Vec3::from_array(*position).is_finite())
    {
        return Err(BoundsError::NonFinite);
    }
    let mut scratch = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    scratch.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    for attribute in [Mesh::ATTRIBUTE_JOINT_INDEX, Mesh::ATTRIBUTE_JOINT_WEIGHT] {
        let data = source
            .attribute(attribute)
            .ok_or(BoundsError::Missing(attribute.name))?;
        scratch.insert_attribute(attribute, data.clone());
    }
    scratch.generate_skinned_mesh_bounds()?;
    let mut envelope = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    envelope.set_skinned_mesh_bounds(scratch.skinned_mesh_bounds().cloned());
    Ok(PartBounds {
        source: id,
        weights: weights.to_vec(),
        envelope,
    })
}

fn update(
    mut commands: Commands,
    meshes: Res<Assets<Mesh>>,
    binds: Res<Assets<SkinnedMeshInverseBindposes>>,
    mut events: MessageReader<AssetEvent<Mesh>>,
    parts: Query<
        (
            Entity,
            &Mesh3d,
            &SkinnedMesh,
            Option<&MeshMorphWeights>,
            Option<&GlobalTransform>,
            Option<&PartBounds>,
        ),
        With<CharacterPart>,
    >,
    morphs: Query<&MorphWeights>,
    joints: Query<&GlobalTransform>,
) {
    let changed = events
        .read()
        .filter_map(|event| match event {
            AssetEvent::Modified { id } => Some(*id),
            _ => None,
        })
        .collect::<HashSet<_>>();
    for (entity, handle, skin, morph, global, cached) in &parts {
        let (Some(source), Some(bind)) =
            (meshes.get(&handle.0), binds.get(&skin.inverse_bindposes))
        else {
            continue;
        };
        let weights = match morph {
            Some(MeshMorphWeights::Value { weights }) => weights.as_slice(),
            Some(MeshMorphWeights::Reference(owner)) => {
                let Ok(weights) = morphs.get(*owner) else {
                    continue;
                };
                weights.weights()
            }
            None => &[],
        };
        let rebuild = cached
            .is_none_or(|cached| cached.source != handle.0.id() || cached.weights != weights)
            || changed.contains(&handle.0.id());
        let rebuilt = if rebuild {
            match prepare(handle.0.id(), source, weights) {
                Ok(bounds) => Some(bounds),
                Err(error) => {
                    error!("Character bounds {entity}: {error}");
                    continue;
                }
            }
        } else {
            None
        };
        let Some(bounds) = rebuilt.as_ref().or(cached) else {
            continue;
        };
        let Ok(aabb) =
            entity_aabb_from_skinned_mesh_bounds(&joints, &bounds.envelope, skin, bind, global)
        else {
            continue;
        };
        commands.entity(entity).insert(Aabb::from(aabb));
        if let Some(bounds) = rebuilt {
            debug!("Character morph/skin bounds prepared for {entity}");
            commands.entity(entity).insert(bounds);
        }
    }
}
