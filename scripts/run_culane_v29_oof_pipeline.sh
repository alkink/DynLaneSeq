#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/DynLaneSeq}"
ROOT="${ROOT:-${REPO_ROOT}/outputs/diagnostics/v29_oof_rbf_gate}"
POLL_SECONDS="${POLL_SECONDS:-300}"

cd "${REPO_ROOT}"

wait_for_artifact() {
  local program="$1"
  local artifact="$2"
  local label="$3"
  local last_notice=0
  while [[ ! -f "${artifact}" ]]; do
    local status
    status="$(supervisorctl status "${program}" 2>&1 || true)"
    if [[ "${status}" != *"RUNNING"* && "${status}" != *"STARTING"* ]]; then
      echo "${label} stopped before its immutable artifact existed: ${status}" >&2
      exit 2
    fi
    local now
    now="$(date +%s)"
    if (( now - last_notice >= 1800 )); then
      echo "Waiting for ${label}: ${status}"
      last_notice="${now}"
    fi
    sleep "${POLL_SECONDS}"
  done
  echo "${label} artifact ready: ${artifact}"
}

SUPPORT_SUMMARY="${ROOT}/supports/support_training_summary.json"
if [[ ! -f "${SUPPORT_SUMMARY}" ]]; then
  support_status="$(supervisorctl status v29_oof_supports 2>&1 || true)"
  if [[ "${support_status}" != *"RUNNING"* ]]; then
    supervisorctl start v29_oof_supports
  fi
  wait_for_artifact v29_oof_supports "${SUPPORT_SUMMARY}" "V29 OOF supports"
fi

BELIEF_LAST="${ROOT}/belief_gate_6k/support_b_to_fold_a/arm_c/training_report.json"
if [[ ! -f "${BELIEF_LAST}" ]]; then
  belief_status="$(supervisorctl status v29_oof_belief_gate 2>&1 || true)"
  if [[ "${belief_status}" != *"RUNNING"* ]]; then
    supervisorctl start v29_oof_belief_gate
  fi
  wait_for_artifact v29_oof_belief_gate "${BELIEF_LAST}" "V29 OOF B/C belief gate"
fi

FINAL_REPORT="${ROOT}/official_val/v29_oof_gate_summary.json"
if [[ ! -f "${FINAL_REPORT}" ]]; then
  eval_status="$(supervisorctl status v29_oof_belief_eval 2>&1 || true)"
  if [[ "${eval_status}" != *"RUNNING"* ]]; then
    supervisorctl start v29_oof_belief_eval
  fi
  wait_for_artifact v29_oof_belief_eval "${FINAL_REPORT}" "V29 OOF validation"
fi

echo "V29 OOF pipeline completed without opening the test set: ${FINAL_REPORT}"
