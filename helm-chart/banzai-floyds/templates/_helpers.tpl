{{/* vim: set filetype=mustache: */}}
{{/*
Expand the name of the chart.
*/}}
{{- define "banzai-floyds.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "banzai-floyds.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "banzai-floyds.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels
*/}}
{{- define "banzai-floyds.labels" -}}
app.kubernetes.io/name: {{ include "banzai-floyds.name" . }}
helm.sh/chart: {{ include "banzai-floyds.chart" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Generate the PostgreSQL DB hostname
*/}}
{{- define "banzai-floyds.dbhost" -}}
{{- if .Values.postgresql.fullnameOverride -}}
{{- .Values.postgresql.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else if .Values.useDockerizedDatabase -}}
{{- printf "%s-postgresql" .Release.Name -}}
{{- else -}}
{{- required "`postgresql.hostname` must be set when `useDockerizedDatabase` is `false`" .Values.postgresql.hostname -}}
{{- end -}}
{{- end -}}

{{/*
Generate the RabbitMQ queue hostname
*/}}
{{- define "banzai-floyds.rabbitmq" -}}
{{- if .Values.rabbitmq.fullnameOverride -}}
{{- .Values.rabbitmq.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else if .Values.useDockerizedRabbitMQ -}}
{{- printf "%s-rabbitmq" .Release.Name -}}
{{- else -}}
{{- required "`rabbitmq.hostname` must be set when `useDockerizedRabbitMQ` is `false`" .Values.rabbitmq.hostname -}}
{{- end -}}
{{- end -}}

{{/*
Define shared environment variables
*/}}
{{- define "banzaiFloyds.Env" -}}
{{/*
External service URLS
*/ -}}
- name: CONFIGDB_URL
  value: {{ .Values.banzaiFloyds.configdbUrl | quote }}
- name: OBSERVATION_PORTAL_URL
  value: {{ .Values.banzaiFloyds.observationPortalUrl | quote }}
{{/*
Ingester environment variables
*/ -}}
- name: API_ROOT
  value: {{ .Values.ingester.apiRoot | quote }}
- name: BUCKET
  value: {{ .Values.ingester.s3Bucket | quote }}
- name: FILESTORE_TYPE
  value: {{ .Values.ingester.filestoreType | quote }} 
{{- if .Values.ingester.noMetrics }}
- name: OPENTSDB_PYTHON_METRICS_TEST_MODE
  value: "1"
{{- else }}
- name: INGESTER_PROCESS_NAME
  value: {{ required "ingester.ingesterProcessName must be defined if noMetrics is false" .Values.ingester.ingesterProcessName | quote }}
- name: OPENTSDB_HOSTNAME
  value: {{ required "ingester.opentsdbHostname must be defined if noMetrics is false" .Values.ingester.opentsdbHostname | quote }}
- name: OPENTSDB_PORT
  value: {{ required "ingester.opentsdbPort must be defined if noMetrics is false" .Values.ingester.opentsdbPort | quote }}
{{- end }}
- name: POSTPROCESS_FILES
{{- if .Values.ingester.postProcessFiles }}
  value: "True"
{{- else }}
  value: "False"
{{- end }}
- name: AUTH_TOKEN
  valueFrom:
    secretKeyRef:
      name: banzai-floyds-secrets
      key: archive-auth-token
- name: AWS_ACCESS_KEY_ID
  valueFrom:
    secretKeyRef:
      name: banzai-floyds-secrets
      key: aws-access-key-id
- name: AWS_SECRET_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: banzai-floyds-secrets
      key: aws-secret-access-key
{{- /*
Optional raw data source environment variables
*/}}
{{- if .Values.banzaiFloyds.useDifferentArchiveSources }}
- name: RAW_DATA_API_ROOT
  value: {{ required "banzaiFloyds.rawDataApiRoot is required if banzaiFloyds.useDifferentArchiveSources is specified." .Values.banzai.rawDataApiRoot | quote }}
- name: RAW_DATA_AUTH_TOKEN
  valueFrom:
    secretKeyRef:
      name: banzai-floyds-secrets
      key: raw-data-auth-token
{{- end -}}
{{/*
BANZAI DB Configuration
*/}}
- name: DB_HOST
  value: {{ include "banzai-floyds.dbhost" . | quote }}
- name: DB_PASSWORD
  valueFrom:
    secretKeyRef:
      name: banzai-floyds-secrets
      key: postgresql-password
- name: DB_USER
  value: {{ .Values.postgresql.postgresqlUsername | quote }}
- name: DB_NAME
  value: {{ .Values.postgresql.postgresqlDatabase | quote }}
- name: DB_ADDRESS
  value: postgresql://$(DB_USER):$(DB_PASSWORD)@$(DB_HOST)/$(DB_NAME)
{{/*
Celery task queue configuration
*/ -}}
- name: RABBITMQ_HOST
  value: {{ include "banzai-floyds.rabbitmq" . | quote }}
- name: RABBITMQ_PASSWORD
  valueFrom:
    secretKeyRef:
      name: banzai-floyds-secrets
      key: rabbitmq-password
- name: TASK_HOST
  value: amqp://{{ .Values.rabbitmq.rabbitmq.username }}:$(RABBITMQ_PASSWORD)@$(RABBITMQ_HOST)/{{ .Values.rabbitmq.vhost }}
- name: RETRY_DELAY
  value: "600"
- name: CALIBRATE_PROPOSAL_ID
  value: {{ .Values.banzaiFloyds.calibrateProposalId | quote }}
- name: FITS_BROKER
  value: {{ .Values.banzaiFloyds.fitsBroker | quote }}
- name: FITS_EXCHANGE
  value: {{ .Values.banzaiFloyds.fitsExchange | quote }}
- name: QUEUE_NAME
  value: {{ .Values.banzaiFloyds.queueName | quote }}
- name: CELERY_TASK_QUEUE_NAME
  value: {{ .Values.banzaiFloyds.celeryTaskQueueName | quote }}
- name: CELERY_LARGE_TASK_QUEUE_NAME
  value: {{ .Values.banzaiFloyds.celeryTaskQueueName | quote }}
- name: BANZAI_WORKER_LOGLEVEL
  value: {{ .Values.banzaiFloyds.banzaiWorkerLogLevel | quote }}
- name: REFERENCE_CATALOG_URL
  value: {{ .Values.banzaiFloyds.PhotometryCatalogURL | quote }}
{{- end -}}
