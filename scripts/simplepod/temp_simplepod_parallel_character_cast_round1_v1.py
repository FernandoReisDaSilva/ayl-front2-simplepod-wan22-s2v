import argparse
import json
import os
import sys
import time
from pathlib import Path

import simplepod_wan22_s2v_runtime_base as base
from simplepod_phase_timing import PhaseTimer, now_iso


TEST_ID = "TEMP_SIMPLEPOD_PARALLEL_CHARACTER_CAST_ROUND1_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "logs" / "character_cast_parallel_round1_summary.json"

IMAGE = "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.2.22-blackwell"
MAX_CONCURRENT_JOBS = "2"
WIDTH = 720
HEIGHT = 720

INPUTS = {
    "alex": {
        "character_id": "alex",
        "taught_language": "EN",
        "output_stem": "alex_en_cast_parallel_round1_720",
        "local_image_path": "data/character_cast/parallel_round1/alex/reference/alex_lipsync_optimized_reference.png",
        "local_audio_path": "data/character_cast/parallel_round1/alex/audio/alex_en_cast_voice_v1_wan15s.wav",
        "input_image_key": "tests/simplepod_character_cast_parallel_round1/inputs/alex/reference/alex_lipsync_optimized_reference.png",
        "input_audio_key": "tests/simplepod_character_cast_parallel_round1/inputs/alex/audio/alex_en_cast_voice_v1_wan15s.wav",
    },
    "sofi": {
        "character_id": "sofi",
        "taught_language": "ES",
        "output_stem": "sofi_es_cast_parallel_round1_720",
        "local_image_path": "data/character_cast/parallel_round1/sofi/reference/sofi_lipsync_optimized_reference.png",
        "local_audio_path": "data/character_cast/parallel_round1/sofi/audio/sofi_es_cast_voice_v1_wan15s.wav",
        "input_image_key": "tests/simplepod_character_cast_parallel_round1/inputs/sofi/reference/sofi_lipsync_optimized_reference.png",
        "input_audio_key": "tests/simplepod_character_cast_parallel_round1/inputs/sofi/audio/sofi_es_cast_voice_v1_wan15s.wav",
    },
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def job_args(spec: dict, args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        test_id=spec["output_stem"],
        character_id=spec["character_id"],
        taught_language=spec["taught_language"],
        width=WIDTH,
        height=HEIGHT,
        output_stem=spec["output_stem"],
        input_image_key=spec["input_image_key"],
        input_audio_key=spec["input_audio_key"],
        local_image_path=spec["local_image_path"],
        local_audio_path=spec["local_audio_path"],
        upload_inputs=True,
        execute=args.execute,
        confirm_start=args.confirm_start,
        confirm_inference=args.confirm_inference,
        confirm_delete=args.confirm_delete,
        instance_market=args.instance_market,
        detail_attempts=args.detail_attempts,
        poll_interval_seconds=args.poll_interval_seconds,
        ready_timeout_seconds=args.ready_timeout_seconds,
        job_timeout_seconds=args.job_timeout_seconds,
        job_poll_interval_seconds=args.job_poll_interval_seconds,
    )


def parallel_runtime_payload(instance_market: str) -> dict:
    payload = base.runtime_payload(instance_market)
    found = False
    for item in payload.get("envVariables", []):
        if item.get("name") == "MAX_CONCURRENT_JOBS":
            item["value"] = MAX_CONCURRENT_JOBS
            found = True
    if not found:
        payload.setdefault("envVariables", []).append({"name": "MAX_CONCURRENT_JOBS", "value": MAX_CONCURRENT_JOBS})
    return payload


def upload_inputs(args: argparse.Namespace, execute_allowed: bool) -> dict:
    results = {}
    for name, spec in INPUTS.items():
        current_args = job_args(spec, args)
        result = base.prepare_input_uploads(current_args, dry_run=not execute_allowed)
        results[name] = result
    return results


def submit_async_job(proxy_url: str, name: str, spec: dict, args: argparse.Namespace) -> dict:
    current_args = job_args(spec, args)
    payload = base.inference_payload(current_args)
    result = base.simple_post(
        proxy_url + base.ASYNC_INFERENCE_ENDPOINT,
        payload,
        timeout_seconds=60,
    )
    body = result.get("json") if isinstance(result.get("json"), dict) else {}
    return {
        "character": name,
        "attempted": True,
        "submitted_at": now_iso(),
        "payload": payload,
        "result": {
            "status": result.get("status"),
            "http_status_code": result.get("http_status_code"),
            "error_type": result.get("error_type", ""),
            "error_truncated": result.get("error_truncated", ""),
            "json": body,
        },
        "accepted": result.get("http_status_code") == 202 and body.get("status") == "accepted",
        "job_id": body.get("job_id") or payload["job_id"],
    }


def terminal_poll_result(job_id: str, elapsed: float, attempts: list[dict], result: dict) -> dict:
    body = result.get("json") if isinstance(result.get("json"), dict) else {}
    return {
        "status": "succeeded" if body.get("status") in {"succeeded", "failed"} else "failed",
        "http_status_code": result.get("http_status_code"),
        "job_id": job_id,
        "elapsed_seconds": round(elapsed, 3),
        "attempts": attempts[-30:],
        "json": body,
        "error_type": result.get("error_type", ""),
        "error_truncated": result.get("error_truncated", ""),
    }


def poll_jobs_independently(proxy_url: str, submissions: dict, args: argparse.Namespace) -> dict:
    started = time.monotonic()
    states = {}
    for name, submission in submissions.items():
        states[name] = {
            "job_id": submission["job_id"],
            "done": not submission.get("accepted"),
            "attempts": [],
            "result": {
                "status": "not_submitted",
                "job_id": submission["job_id"],
                "json": submission.get("result", {}).get("json", {}),
            },
        }
        if not submission.get("accepted"):
            states[name]["result"] = {
                "status": "submit_failed",
                "job_id": submission["job_id"],
                "json": submission.get("result", {}).get("json", {}),
                "http_status_code": submission.get("result", {}).get("http_status_code"),
                "error_type": submission.get("result", {}).get("error_type", ""),
                "error_truncated": submission.get("result", {}).get("error_truncated", ""),
            }

    while not all(state["done"] for state in states.values()):
        elapsed = time.monotonic() - started
        if elapsed > args.job_timeout_seconds:
            for name, state in states.items():
                if state["done"]:
                    continue
                state["done"] = True
                state["result"] = {
                    "status": "timeout",
                    "job_id": state["job_id"],
                    "elapsed_seconds": round(elapsed, 3),
                    "attempts": state["attempts"][-30:],
                    "error_type": "JobTimeout",
                    "error_truncated": f"Async job did not finish within {args.job_timeout_seconds}s.",
                }
            break

        for name, state in states.items():
            if state["done"]:
                continue
            result = base.simple_get(proxy_url + f"/admin/jobs/{state['job_id']}", timeout_seconds=30)
            body = result.get("json") if isinstance(result.get("json"), dict) else {}
            job_status = body.get("status") or result.get("status")
            state["attempts"].append(
                {
                    "elapsed_seconds": round(elapsed, 3),
                    "http_status_code": result.get("http_status_code"),
                    "job_status": job_status,
                    "error_type": result.get("error_type", ""),
                }
            )
            print(f"[{TEST_ID}] {name}_job_status={job_status} elapsed={int(elapsed)}s", flush=True)
            if body.get("status") in {"succeeded", "failed"}:
                state["done"] = True
                state["result"] = terminal_poll_result(state["job_id"], elapsed, state["attempts"], result)
            elif result.get("status") not in {"succeeded", "timeout"} and result.get("http_status_code") != 404:
                state["done"] = True
                state["result"] = terminal_poll_result(state["job_id"], elapsed, state["attempts"], result)

        if not all(state["done"] for state in states.values()):
            time.sleep(max(1, args.job_poll_interval_seconds))

    return {name: state["result"] for name, state in states.items()}


def final_report_json(poll_result: dict) -> dict:
    value = poll_result.get("json")
    if not isinstance(value, dict):
        return {}
    summary = value.get("summary")
    return summary if isinstance(summary, dict) else value


def summary_value(report: dict, *keys: str):
    for key in keys:
        if key in report:
            return report.get(key)
    return None


def download_result_for_job(name: str, spec: dict, args: argparse.Namespace, dry_run: bool) -> dict:
    current_args = job_args(spec, args)
    video_key = base.output_video_key(current_args)
    report_key = base.output_report_key(current_args)
    output_dir = REPO_ROOT / "data" / "character_cast" / "parallel_round1" / name / "outputs"
    results = {
        "video": {
            "key": video_key,
            "local_path": str(output_dir / f"{spec['output_stem']}.mp4"),
            "attempted": not dry_run,
            "status": "planned" if dry_run else "pending",
        },
        "report": {
            "key": report_key,
            "local_path": str(output_dir / f"{spec['output_stem']}_final_report.json"),
            "attempted": not dry_run,
            "status": "planned" if dry_run else "pending",
        },
    }
    if dry_run:
        return results

    try:
        import boto3

        config = base.r2_client_config()
        client = boto3.client(
            "s3",
            endpoint_url=config["endpoint_url"],
            aws_access_key_id=config["access_key_id"],
            aws_secret_access_key=config["secret_access_key"],
            region_name=config["region"] or "auto",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        bucket = config["bucket"]
        for item in results.values():
            client.download_file(bucket, item["key"], item["local_path"])
            item["status"] = "succeeded"
    except Exception as exc:
        for item in results.values():
            if item["status"] != "succeeded":
                item["status"] = "failed"
                item["error_type"] = type(exc).__name__
                item["error_truncated"] = str(exc)[:1000]
    return results


def build_report(args: argparse.Namespace, status: str, data: dict, error: str = "") -> dict:
    alex_report = final_report_json(data.get("poll_results", {}).get("alex", {}))
    sofi_report = final_report_json(data.get("poll_results", {}).get("sofi", {}))
    alex_peak = summary_value(alex_report, "peak_vram_gb")
    sofi_peak = summary_value(sofi_report, "peak_vram_gb")
    observed_peaks = [value for value in (alex_peak, sofi_peak) if isinstance(value, (int, float))]
    attention_backend = summary_value(alex_report, "attention_backend_used", "attention_backend")
    attention_fallback = summary_value(alex_report, "attention_fallback_applied", "attention_fallback")
    if attention_backend is None:
        attention_backend = summary_value(sofi_report, "attention_backend_used", "attention_backend")
    if attention_fallback is None:
        attention_fallback = summary_value(sofi_report, "attention_fallback_applied", "attention_fallback")
    delete_result = data.get("delete_result", {})

    return {
        "script_id": TEST_ID,
        "created_at": now_iso(),
        "status": status,
        "error": error,
        "dry_run": not (args.execute and args.confirm_start and args.confirm_inference and args.confirm_delete),
        "pod_start_seconds": data.get("pod_start_seconds"),
        "ready_seconds": data.get("ready_seconds"),
        "wall_clock_seconds": data.get("wall_clock_seconds"),
        "alex_runtime_seconds": summary_value(alex_report, "runtime_seconds"),
        "sofi_runtime_seconds": summary_value(sofi_report, "runtime_seconds"),
        "alex_peak_vram_gb": alex_peak,
        "sofi_peak_vram_gb": sofi_peak,
        "max_peak_vram_observed": max(observed_peaks) if observed_peaks else None,
        "alex_status": summary_value(alex_report, "job_status", "status") or data.get("poll_results", {}).get("alex", {}).get("status"),
        "sofi_status": summary_value(sofi_report, "job_status", "status") or data.get("poll_results", {}).get("sofi", {}).get("status"),
        "attention_backend": attention_backend,
        "attention_fallback": attention_fallback,
        "delete_status": delete_result.get("status"),
        "image_ref": IMAGE,
        "template_id": base.TEMPLATE_ID,
        "max_concurrent_jobs": int(MAX_CONCURRENT_JOBS),
        "input_uploads": data.get("input_uploads", {}),
        "submissions": data.get("submissions", {}),
        "poll_results": data.get("poll_results", {}),
        "downloads": data.get("downloads", {}),
        "instance_id": data.get("instance_id"),
        "public_api_base_url": data.get("public_api_base_url", ""),
        "market_selection": data.get("market_selection", {}),
        "runtime_verify_result": data.get("runtime_verify_result", {}),
        "gpu_check": data.get("gpu_check", {}),
        "delete_result": delete_result,
        "phase_timings": data.get("phase_timings", []),
        "safety_guards": {
            "created_new_script_only": True,
            "simplepod_start_called": bool(data.get("start_result", {}).get("attempted")),
            "r2_upload_attempted": any(
                item.get("uploaded_image_result", {}).get("attempted")
                or item.get("uploaded_audio_result", {}).get("attempted")
                for item in data.get("input_uploads", {}).values()
                if isinstance(item, dict)
            ),
            "inference_jobs_submitted": bool(data.get("submissions")),
            "delete_attempted": bool(delete_result.get("attempted")),
            "parallel_jobs": True,
            "secrets_printed": False,
        },
        "runtime": data,
    }


def blocked_status(args: argparse.Namespace) -> str:
    for spec in INPUTS.values():
        if args.execute and not base.local_path_value(spec["local_image_path"]).is_file():
            return "blocked_missing_local_image"
        if args.execute and not base.local_path_value(spec["local_audio_path"]).is_file():
            return "blocked_missing_local_audio"
    if args.execute and not args.confirm_start:
        return "blocked_missing_confirm_start"
    if args.execute and not args.confirm_inference:
        return "blocked_missing_confirm_inference"
    if args.execute and not args.confirm_delete:
        return "blocked_missing_confirm_delete"
    return ""


def delete_instance(base_url: str, api_key: str, instance_id: int | None, data: dict, timer: PhaseTimer) -> None:
    if instance_id is None:
        data["delete_result"] = {
            "attempted": False,
            "status": "skipped_no_instance_id",
        }
        return
    print(f"[{TEST_ID}] cleanup_started pod_id={instance_id}", flush=True)
    with timer.phase("delete_instance"):
        delete_result = base.smoke.http_request(
            base_url,
            base.smoke.DELETE_INSTANCE_PATH.format(id=instance_id),
            api_key,
            method="DELETE",
        )
    data["delete_result"] = {
        key: delete_result.get(key)
        for key in ("attempted", "status", "method", "path", "http_status_code", "endpoint_host", "error_type", "error_truncated")
    }
    data["manual_cleanup_required"] = data["delete_result"].get("http_status_code") not in {200, 202, 204}
    print(
        f"[{TEST_ID}] cleanup_completed delete_status={data['delete_result'].get('status')} "
        f"manual_cleanup_required={str(data['manual_cleanup_required']).lower()}",
        flush=True,
    )


def run(args: argparse.Namespace) -> int:
    data = {
        "delete_result": {
            "attempted": False,
            "status": "not_started",
        },
        "manual_cleanup_required": False,
    }
    timer = PhaseTimer()
    data["phase_timings"] = timer.phases
    started_monotonic = time.monotonic()
    instance_id = None
    api_key = ""
    base_url = base.smoke.DEFAULT_BASE_URL
    status = "failed"
    try:
        execute_allowed = args.execute and args.confirm_start and args.confirm_inference and args.confirm_delete
        print(f"[{TEST_ID}] START test_id={TEST_ID} dry_run={str(not execute_allowed).lower()}", flush=True)
        print(
            f"[{TEST_ID}] image={IMAGE} gpu=RTX_PRO_6000_Blackwell_96GB "
            f"timeout={args.job_timeout_seconds}s execute={str(execute_allowed).lower()}",
            flush=True,
        )

        blocked = blocked_status(args)
        if blocked:
            status = blocked
            data["wall_clock_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} manual_cleanup_required=false report={REPORT_PATH}", flush=True)
            return 1

        with timer.phase("load_auth_env"):
            base.load_local_env()
            api_key = os.getenv(base.smoke.API_KEY_ENV, "")
            base_url = os.getenv(base.smoke.BASE_URL_ENV, base.smoke.DEFAULT_BASE_URL)
            data["r2_env_local_check"] = base.local_r2_env_presence()

        with timer.phase("prepare_input_uploads"):
            data["input_uploads"] = upload_inputs(args, execute_allowed)

        if execute_allowed:
            uploads_ok = all(
                result.get("uploaded_image_result", {}).get("status") == "succeeded"
                and result.get("uploaded_audio_result", {}).get("status") == "succeeded"
                for result in data["input_uploads"].values()
            )
            if not uploads_ok:
                status = "failed_upload_inputs"
                data["wall_clock_seconds"] = round(time.monotonic() - started_monotonic, 3)
                write_json(REPORT_PATH, build_report(args, status, data))
                print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}", flush=True)
                return 1

        if api_key:
            with timer.phase("market_selection"):
                market = base.choose_market(args, base_url, api_key, data)
        else:
            market = args.instance_market
            data["market_selection"] = {
                "status": "skipped_missing_api_key",
                "selected": base.select_full_blackwell_market([]),
            }

        if not execute_allowed:
            status = "dry_run_ready"
            data["instance_payload_dryrun"] = base.redact_instance_payload(
                parallel_runtime_payload(args.instance_market or "<selected_full_blackwell_96gb_market>")
            )
            data["wall_clock_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}", flush=True)
            return 0

        if not api_key:
            status = "missing_api_key"
            data["wall_clock_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}", flush=True)
            return 1
        missing_r2 = base.missing_local_r2_env()
        if missing_r2:
            status = "missing_local_r2_env"
            data["missing_local_r2_env"] = missing_r2
            data["wall_clock_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}", flush=True)
            return 1
        if not market:
            status = "blocked_no_full_blackwell_96gb_market_selected"
            data["wall_clock_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}", flush=True)
            return 1

        print(f"[{TEST_ID}] pod_creation_requested", flush=True)
        pod_start_monotonic = time.monotonic()
        with timer.phase("start_instance"):
            start_payload = parallel_runtime_payload(market)
            start_result = base.smoke.http_request(
                base_url,
                base.smoke.START_INSTANCE_PATH,
                api_key,
                method="POST",
                payload=start_payload,
            )
        data["pod_start_seconds"] = round(time.monotonic() - pod_start_monotonic, 3)
        data["start_result"] = {
            key: start_result.get(key)
            for key in (
                "attempted",
                "status",
                "method",
                "path",
                "http_status_code",
                "endpoint_host",
                "content_type",
                "body_bytes",
                "error_type",
                "error_truncated",
                "response_body_truncated",
            )
        }
        data["start_result"]["request_payload_redacted"] = base.redact_instance_payload(start_payload)
        data["start_result"]["json"] = start_result.get("json")
        instance_id = base.smoke.extract_instance_id(start_result.get("json"))
        data["instance_id"] = instance_id
        print(f"[{TEST_ID}] pod_id={instance_id}", flush=True)
        if start_result.get("status") != "succeeded" or instance_id is None:
            status = "start_failed"
            return 1

        ready_monotonic = time.monotonic()
        with timer.phase("wait_public_url"):
            proxy_url = base.wait_for_public_url(base_url, api_key, instance_id, args, data)
        data["public_api_base_url"] = proxy_url
        if not proxy_url:
            status = "blocked_no_proxy_url_for_port_8000"
            return 1

        with timer.phase("wait_health"):
            readiness, readiness_attempts, _ = base.smoke.wait_for_instance_api(proxy_url, args.ready_timeout_seconds)
        data["ready_seconds"] = round(time.monotonic() - ready_monotonic, 3)
        data["api_readiness"] = {"status": readiness, "attempts": readiness_attempts}
        if readiness != "ready":
            status = "api_not_ready"
            return 1

        with timer.phase("gpu_check"):
            gpu_result = base.smoke.simple_get(proxy_url + "/gpu")
        data["gpu_check"] = {
            "status": gpu_result.get("status"),
            "http_status_code": gpu_result.get("http_status_code"),
            "summary": base.blackwell_smoke.summarize_gpu(gpu_result.get("json")),
            "full_blackwell_runtime_check": base.runtime_gpu_is_full_blackwell(gpu_result.get("json")),
        }
        if data["gpu_check"]["full_blackwell_runtime_check"].get("status") != "passed":
            status = "blocked_runtime_not_full_blackwell_96gb_or_mig"
            return 1

        with timer.phase("wan22_runtime_verify"):
            verify_result = base.smoke.simple_get(proxy_url + base.VERIFY_ENDPOINT)
        data["runtime_verify_result"] = {
            "attempted": True,
            "status": verify_result.get("status"),
            "http_status_code": verify_result.get("http_status_code"),
            "error_type": verify_result.get("error_type", ""),
            "error_truncated": verify_result.get("error_truncated", ""),
            "summary": base.summarize_runtime_verify(verify_result.get("json")),
        }
        if not base.runtime_verify_passed(verify_result):
            status = "failed_wan22_runtime_verify_before_inference"
            return 1

        with timer.phase("submit_async_jobs"):
            data["submissions"] = {
                name: submit_async_job(proxy_url, name, spec, args)
                for name, spec in INPUTS.items()
            }
        if not all(item.get("accepted") for item in data["submissions"].values()):
            status = "failed_submit_async_jobs"
        else:
            print(f"[{TEST_ID}] wait_for_r2_progress_started jobs=alex,sofi", flush=True)
            with timer.phase("poll_async_jobs"):
                data["poll_results"] = poll_jobs_independently(proxy_url, data["submissions"], args)
            with timer.phase("download_outputs"):
                data["downloads"] = {
                    name: download_result_for_job(name, spec, args, dry_run=False)
                    for name, spec in INPUTS.items()
                }
            status = "succeeded" if all(
                final_report_json(result).get("status") in {"succeeded", "success", None}
                and result.get("status") == "succeeded"
                for result in data.get("poll_results", {}).values()
            ) else "completed_with_job_failure"

        return 0 if status == "succeeded" else 1
    except Exception as exc:
        status = "failed"
        error = str(exc)
        data["wall_clock_seconds"] = round(time.monotonic() - started_monotonic, 3)
        write_json(REPORT_PATH, build_report(args, status, data, error))
        print(f"[{TEST_ID}] ERROR {error[:300]}", file=sys.stderr, flush=True)
        return 1
    finally:
        if args.execute and args.confirm_delete and api_key:
            delete_instance(base_url, api_key, instance_id, data, timer)
        elif instance_id is not None:
            data["manual_cleanup_required"] = True
        data["wall_clock_seconds"] = round(time.monotonic() - started_monotonic, 3)
        delete_status = data.get("delete_result", {}).get("status")
        if status == "succeeded" and delete_status not in {"succeeded"}:
            status = "delete_failed"
        write_json(REPORT_PATH, build_report(args, status, data))
        print(
            f"[{TEST_ID}] DONE status={status} delete_status={delete_status} "
            f"manual_cleanup_required={str(data.get('manual_cleanup_required', False)).lower()} report={REPORT_PATH}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental parallel Alex/Sofi Wan2.2 S2V test on one SimplePod Blackwell 96GB instance."
    )
    parser.add_argument("--execute", action="store_true", help="Create SimplePod, upload inputs, submit jobs, download outputs, then delete.")
    parser.add_argument("--confirm-start", action="store_true", help="Required with --execute to create the SimplePod instance.")
    parser.add_argument("--confirm-inference", action="store_true", help="Required with --execute to submit both inference jobs.")
    parser.add_argument("--confirm-delete", action="store_true", help="Required with --execute to delete the instance at the end.")
    parser.add_argument("--instance-market", default="", help="Optional explicit /instances/market/{id}; otherwise select EU-PL-01 full Blackwell 96GB.")
    parser.add_argument("--detail-attempts", type=int, default=60)
    parser.add_argument("--poll-interval-seconds", type=int, default=5)
    parser.add_argument("--ready-timeout-seconds", type=int, default=900)
    parser.add_argument("--job-timeout-seconds", type=int, default=3600)
    parser.add_argument("--job-poll-interval-seconds", type=int, default=15)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
