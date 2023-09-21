project_id      = {project_id}
region          = {region}
resource_prefix = {cluster_prefix}
metadata        = {metadata}
labels          = {labels}
disk_size_gb    = {disk_size_gb}
machine_image   = {
  "family": {image_family},
  "name": {image_name},
  "project": {image_project}
}
instance_groups = [
  {
    target_size = {node_count}
    zone        = {nodes_zone}
    machine_type= {machine_type}
  }
]
