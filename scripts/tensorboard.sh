set -x
export REPORT_IP=$(ip addr show eth1 | grep -w inet | awk '{print $2}' | awk -F '/' '{print $1}')
export REPORT_SERVER_TYPE=tensorboard
export REPORT_PORT=54651
export REPORT_TOKEN=token
report_url="${ENV_REPORT_URL}?signed_rtx=${ENV_SIGN_RTX}&insName=${ENV_APP_INS_NAME}&routeIpportInfo=${REPORT_SERVER_TYPE}|${REPORT_IP}:${REPORT_PORT}&token=${REPORT_TOKEN}"
curl "$report_url"
export LOG_DIR=$1

tensorboard --logdir $LOG_DIR --port $REPORT_PORT --reload_multifile true --bind_all