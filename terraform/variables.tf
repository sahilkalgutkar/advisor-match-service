variable "project_id" {
  description = "GCP project id"
  type        = string
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "elasticsearch_url" {
  description = "URL of the Elasticsearch cluster the service connects to (a managed Elastic Cloud deployment in a real environment - this repo doesn't provision Elasticsearch itself)"
  type        = string
}
