#!/usr/bin/env groovy
@Library('lco-shared-libs@0.1.3') _

pipeline {
    agent any
	parameters {
		booleanParam(
			name: 'forceEndToEnd',
			defaultValue: false,
			description: 'When true, forces the end-to-end tests to always run.')
	}
	environment {
		dockerImage = null
		PROJ_NAME = projName()
		GIT_DESCRIPTION = sh(script: 'git describe', returnStdout: true).trim()
		DOCKER_IMG = dockerImageName("${LCO_DOCK_REG}", "${PROJ_NAME}", "${GIT_DESCRIPTION}")
	}
	options {
		timeout(time: 8, unit: 'HOURS')
		lock resource: 'BANZAIFLOYDSLock'
	}
	stages {
		stage('Build image') {
			steps {
				script {
					dockerImage = docker.build("${DOCKER_IMG}", "--pull .")
				}
			}
		}
		stage('Push image') {
			steps {
				script {
					dockerImage.push("${GIT_DESCRIPTION}")
				}
			}
		}
		stage('DeployTestStack') {
			agent {
				label 'helm'
			}
			when {
				anyOf {
					branch 'PR-*'
					branch 'dev'
					expression { return params.forceEndToEnd }
				}
			}
		    steps {
	            script {
                    withKubeConfig([credentialsId: "build-kube-config"]) {
                        sh('helm repo update')
                        final cmd = " helm3 --namespace build delete banzai-floyds-e2e &> cleanup.txt"
                        final status = sh(script: cmd, returnStatus: true)
                        final output = readFile('cleanup.txt').trim()
                        sh(script: "rm -f cleanup.txt", returnStatus: true)
                        echo output
                        sh('helm3 --namespace build upgrade --install banzai-floyds-e2e helm-chart/banzai-floyds-e2e ' +
                            '--set image.tag="${GIT_DESCRIPTION}" --force --wait --timeout=3600s')

                        podName = sh(script: 'kubectl get po -l app.kubernetes.io/instance=banzai-floyds-e2e ' +
                                        '--sort-by=.status.startTime -o jsonpath="{.items[-1].metadata.name}"',
                                     returnStdout: true).trim()

                    }
                 }
		    }
		}
		stage('Test-Order-Detection') {
			environment {
				// store stage start time in the environment so it has stage scope
				START_TIME = sh(script: 'date +%s', returnStdout: true).trim()
			}
			when {
				anyOf {
					branch 'PR-*'
					expression { return params.forceEndToEnd }
				}
			}
			steps {
				script {
                    withKubeConfig([credentialsId: "build-kube-config"]) {
						sh("kubectl exec ${podName} -c banzai-floyds-e2e-listener -- " +
						        "pytest -s --durations=0 --junitxml=/home/archive/pytest-order-detection.xml " +
						        "-m detect_orders /lco/banzai-floyds/")
					}
				}
			}
			post {
				always {
					script {
					    withKubeConfig([credentialsId: "build-kube-config"]) {
					    	env.LOGS_SINCE = sh(script: 'expr `date +%s` - ${START_TIME}', returnStdout: true).trim()
    					    sh("kubectl logs ${podName} --since=${LOGS_SINCE}s --all-containers")
						    sh("kubectl cp -c banzai-floyds-e2e-listener ${podName}:/home/archive/pytest-order-detection.xml " +
						            "pytest-order-detection.xml")
						    junit "pytest-order-detection.xml"
						}
					}
				}
			}
		}
		stage('Test-Arc-Frame-Creation') {
			agent {
				label 'helm'
			}
			environment {
				// store stage start time in the environment so it has stage scope
				START_TIME = sh(script: 'date +%s', returnStdout: true).trim()
			}
			when {
				anyOf {
					branch 'PR-*'
					expression { return params.forceEndToEnd }
				}
			}
			steps {
				script {
                    withKubeConfig([credentialsId: "build-kube-config"]) {
						sh("kubectl exec ${podName} -c banzai-floyds-e2e-listener -- " +
						        "pytest -s --durations=0 --junitxml=/home/archive/pytest-arc-frames.xml " +
						        "-m arc_frames /lco/banzai-floyds/")
					}
				}
			}
			post {
				always {
					script {
					    withKubeConfig([credentialsId: "build-kube-config"]) {
					    	env.LOGS_SINCE = sh(script: 'expr `date +%s` - ${START_TIME}', returnStdout: true).trim()
    					    sh("kubectl logs ${podName} --since=${LOGS_SINCE}s --all-containers")
						    sh("kubectl cp -c banzai-floyds-e2e-listener ${podName}:/home/archive/pytest-arc-frames.xml " +
						            "pytest-arc-frames.xml")
						    junit "pytest-arc-frames.xml"
						}
					}
				}
			}
		}
		stage('Test-Science-Frame-Creation') {
			agent {
				label 'helm'
			}
			environment {
				// store stage start time in the environment so it has stage scope
				START_TIME = sh(script: 'date +%s', returnStdout: true).trim()
			}
			when {
				anyOf {
					branch 'PR-*'
					expression { return params.forceEndToEnd }
				}
			}
			steps {
				script {
                    withKubeConfig([credentialsId: "build-kube-config"]) {
						sh("kubectl exec ${podName} -c banzai-floyds-e2e-listener -- " +
						        "pytest -s --durations=0 --junitxml=/home/archive/pytest-science-frames.xml " +
						        "-m science_frames /lco/banzai-floyds/")
					}
				}
			}
			post {
				always {
					script {
					    withKubeConfig([credentialsId: "build-kube-config"]) {
					    	env.LOGS_SINCE = sh(script: 'expr `date +%s` - ${START_TIME}', returnStdout: true).trim()
    					    sh("kubectl logs ${podName} --since=${LOGS_SINCE}s --all-containers")
						    sh("kubectl cp -c banzai-floyds-e2e-listener ${podName}:/home/archive/pytest-science-frames.xml " +
						            "pytest-science-frames.xml")
						    junit "pytest-science-frames.xml"
						}
					}
				}
				success {
					script {
					    withKubeConfig([credentialsId: "build-kube-config"]) {
                            sh("helm3 --namespace build delete banzai-floyds-e2e || true")
					    }
					}
				}
			}
		}
    }
}
