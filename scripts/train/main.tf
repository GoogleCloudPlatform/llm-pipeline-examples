variable "instance_groups" {}
variable "project_id" {}
variable "region" {}
variable "resource_prefix" {}
variable "metadata" {}
variable "labels" {}
variable "startup_script" {}
variable "machine_image" {}
variable "disk_size_gb" {}
variable "maintenance_interval" {}
variable "use_compact_placement_policy" {}


module "mig-cos" {
  source = "github.com/GoogleCloudPlatform/ai-infra-cluster-provisioning//a3/terraform/modules/cluster/mig-cos"

  instance_groups               = var.instance_groups
  project_id                    = var.project_id
  region                        = var.region
  resource_prefix               = var.resource_prefix
  metadata                      = var.metadata
  labels                        = var.labels
  startup_script                = var.startup_script
  machine_image                 = var.machine_image
  disk_size_gb                  = var.disk_size_gb
  maintenance_interval          = var.maintenance_interval
  use_compact_placement_policy  = var.use_compact_placement_policy
}
