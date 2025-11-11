import os
import datetime
import asyncio
import subprocess
import sys
import time
import psutil
from typing import List, Dict, Any, Optional

# Lazy import: TeethTemplateMaker는 실제 사용 시점에 import
# from single_template_maker_lib import TeethTemplateMaker

class GingivaGenerator:
    """
    치은(Gingiva) 생성 클래스
    
    TeethTemplateMaker를 사용하여 치아 입력 데이터로부터 치은 메쉬를 생성합니다.
    """
    
    def __init__(self, websocket=None, use_cpu_affinity=False):
        """
        GingivaGenerator 초기화
        
        Args:
            websocket: WebSocket 연결 객체 (선택적, 진행 상황 알림용)
            use_cpu_affinity: CPU Affinity 최적화 사용 여부 (기본값: False)
        """
        self.websocket = websocket
        self.use_cpu_affinity = use_cpu_affinity
        self.cpu_affinity_allocation = None
        
        if use_cpu_affinity:
            self._init_cpu_affinity()
    
    def _init_cpu_affinity(self):
        """CPU Affinity 최적화 초기화"""
        try:
            cpu_count = psutil.cpu_count(logical=False)
            cpu_count_logical = psutil.cpu_count(logical=True)
            
            if cpu_count is None or cpu_count < 8:
                print(f"[CPU AFFINITY] CPU 코어 부족 ({cpu_count}개) - 최적화 비활성화")
                self.use_cpu_affinity = False
                return
            
            # CPU 코어를 반으로 나누어 할당
            mid = cpu_count_logical // 2
            self.cpu_affinity_allocation = {
                "mandibular": list(range(0, mid)),
                "maxilla": list(range(mid, cpu_count_logical))
            }
            
            print(f"[CPU AFFINITY] 최적화 활성화")
            print(f"[CPU AFFINITY]   mandibular: 코어 {self.cpu_affinity_allocation['mandibular']}")
            print(f"[CPU AFFINITY]   maxilla: 코어 {self.cpu_affinity_allocation['maxilla']}")
            
        except Exception as e:
            print(f"[CPU AFFINITY] 초기화 실패: {e}")
            self.use_cpu_affinity = False

    
    async def generate_gingiva_parallel(
        self,
        input_path: str,
        output_path: str,
        arch_types: List[str],
        request_id: str
    ) -> Dict[str, Any]:
        """
        치은 생성 실행 (병렬 처리)
        
        각 arch_type별로 별도의 프로세스를 생성하여 병렬로 처리합니다.
        성능: 2개의 arch_type을 처리할 때 순차 처리 대비 약 50% 시간 단축
        
        Args:
            input_path: 치아 입력 파일들이 있는 경로
            output_path: 생성된 치은 파일을 저장할 경로
            arch_types: 생성할 치은 타입 리스트 ["maxilla", "mandibular"]
            request_id: 요청 추적을 위한 고유 ID
            
        Returns:
            Dict: 생성 결과 정보
            {
                "status": "success" | "error",
                "generated_files": [{"arch_type": str, "file_path": str}],
                "error": str (오류 발생 시)
            }
        """
        # === 성능 프로파일링 시작 ===
        total_start_time = time.perf_counter()
        
        try:
            print(f"[{request_id}] 치은 생성 시작 (병렬 처리 모드)")
            print(f"[{request_id}] input_path: {input_path}")
            print(f"[{request_id}] output_path: {output_path}")
            print(f"[{request_id}] arch_types: {arch_types}")
            
            # 출력 디렉토리 생성
            dir_start = time.perf_counter()
            os.makedirs(output_path, exist_ok=True)
            dir_elapsed = time.perf_counter() - dir_start
            print(f"[{request_id}] [PERF] 출력 디렉토리 생성: {dir_elapsed:.4f}초")
            print(f"[{request_id}] 출력 디렉토리 생성 완료")
            
            # WebSocket으로 시작 메시지 전송
            ws_start = time.perf_counter()
            await self._send_websocket_message({
                "type": "gingiva_generation_started",
                "request_id": request_id,
                "message": f"치은 생성이 시작되었습니다 (병렬 처리: {len(arch_types)}개).",
                "arch_types": arch_types,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            ws_elapsed = time.perf_counter() - ws_start
            print(f"[{request_id}] [PERF] WebSocket 시작 메시지 전송: {ws_elapsed:.4f}초")
            
            # 각 arch_type별로 별도의 프로세스 생성 및 병렬 실행
            task_setup_start = time.perf_counter()
            tasks = []
            for arch_type in arch_types:
                task = self._generate_single_arch_type(
                    input_path,
                    output_path,
                    arch_type,
                    request_id
                )
                tasks.append(task)
            task_setup_elapsed = time.perf_counter() - task_setup_start
            print(f"[{request_id}] [PERF] Task 설정: {task_setup_elapsed:.4f}초")
            
            print(f"[{request_id}] {len(tasks)}개의 프로세스를 병렬로 실행합니다...")
            
            # 모든 프로세스를 병렬로 실행
            parallel_start = time.perf_counter()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            parallel_elapsed = time.perf_counter() - parallel_start
            print(f"[{request_id}] [PERF] 병렬 처리 완료: {parallel_elapsed:.4f}초")
            
            # 결과 취합
            result_process_start = time.perf_counter()
            generated_files = []
            errors = []
            
            for i, result in enumerate(results):
                arch_type = arch_types[i]
                
                if isinstance(result, Exception):
                    error_msg = f"{arch_type}: {str(result)}"
                    print(f"[{request_id}] 오류 발생: {error_msg}")
                    errors.append(error_msg)
                elif result["status"] == "success":
                    generated_files.extend(result["generated_files"])
                else:
                    errors.append(f"{arch_type}: {result.get('error', 'Unknown error')}")
            
            result_process_elapsed = time.perf_counter() - result_process_start
            print(f"[{request_id}] [PERF] 결과 취합: {result_process_elapsed:.4f}초")
            
            # 결과 판단
            if not generated_files:
                error_msg = "치은 파일이 생성되지 않음. " + "; ".join(errors)
                print(f"[{request_id}] 치은 생성 실패: {error_msg}")
                raise Exception(error_msg)
            
            print(f"[{request_id}] 치은 생성 완료 (파일 {len(generated_files)}개 생성)")
            
            # 일부 실패한 경우 경고
            if errors:
                print(f"[{request_id}] 경고: 일부 arch_type 처리 실패: {errors}")
            
            # WebSocket으로 완료 메시지 전송
            ws_complete_start = time.perf_counter()
            await self._send_websocket_message({
                "type": "gingiva_generation_completed",
                "request_id": request_id,
                "generated_files": generated_files,
                "warnings": errors if errors else None,
                "message": f"치은 생성이 완료되었습니다 ({len(generated_files)}개 생성).",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            ws_complete_elapsed = time.perf_counter() - ws_complete_start
            print(f"[{request_id}] [PERF] WebSocket 완료 메시지 전송: {ws_complete_elapsed:.4f}초")
            
            # === 전체 성능 로깅 ===
            total_elapsed = time.perf_counter() - total_start_time
            print(f"[{request_id}] [PERF] ========================================")
            print(f"[{request_id}] [PERF] 전체 실행 시간: {total_elapsed:.4f}초 ({total_elapsed/60:.2f}분)")
            print(f"[{request_id}] [PERF] arch_type 개수: {len(arch_types)}")
            print(f"[{request_id}] [PERF] 생성된 파일 수: {len(generated_files)}")
            print(f"[{request_id}] [PERF] ========================================")
            
            return {
                "status": "success",
                "generated_files": generated_files,
                "warnings": errors if errors else None,
                "performance": {
                    "total_time": total_elapsed,
                    "parallel_processing_time": parallel_elapsed,
                    "arch_types_count": len(arch_types),
                    "files_generated": len(generated_files)
                }
            }
            
        except Exception as e:
            total_elapsed = time.perf_counter() - total_start_time
            print(f"[{request_id}] [PERF] 오류 발생 전까지 실행 시간: {total_elapsed:.4f}초")
            print(f"[{request_id}] 치은 생성 중 오류 발생: {str(e)}")
            
            # WebSocket으로 오류 메시지 전송
            await self._send_websocket_message({
                "type": "gingiva_generation_failed",
                "request_id": request_id,
                "error": str(e),
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _generate_single_arch_type(
        self,
        input_path: str,
        output_path: str,
        arch_type: str,
        request_id: str
    ) -> Dict[str, Any]:
        """
        단일 arch_type에 대한 치은 생성 (별도 프로세스)
        
        Args:
            input_path: 치아 입력 파일들이 있는 경로
            output_path: 생성된 치은 파일을 저장할 경로
            arch_type: 생성할 치은 타입 ("maxilla" 또는 "mandibular")
            request_id: 요청 추적을 위한 고유 ID
            
        Returns:
            Dict: 생성 결과 정보
        """
        # === 단일 arch_type 성능 프로파일링 시작 ===
        arch_start_time = time.perf_counter()
        
        try:
            print(f"[{request_id}] [{arch_type}] 프로세스 시작")
            
            # 별도 프로세스에서 단일 arch_type 처리
            import json
            setup_start = time.perf_counter()
            script_path = os.path.join(os.path.dirname(__file__), "run_gingiva_generation.py")
            arch_types_json = json.dumps([arch_type])  # 단일 arch_type을 리스트로 전달
            
            # subprocess를 비동기로 실행
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['PYTHONIOENCODING'] = 'utf-8'
            
            setup_elapsed = time.perf_counter() - setup_start
            print(f"[{request_id}] [{arch_type}] [PERF] 프로세스 준비: {setup_elapsed:.4f}초")
            
            # CPU Affinity 정보 전달
            cmd_args = [
                sys.executable,
                "-u",
                script_path,
                input_path,
                output_path,
                arch_types_json
            ]
            
            if self.use_cpu_affinity and self.cpu_affinity_allocation:
                cpu_affinity_json = json.dumps(self.cpu_affinity_allocation)
                cmd_args.append(cpu_affinity_json)
                print(f"[{request_id}] [{arch_type}] [CPU AFFINITY] 코어 할당: {self.cpu_affinity_allocation.get(arch_type, 'N/A')}")
            
            process_start = time.perf_counter()
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            process_create_elapsed = time.perf_counter() - process_start
            print(f"[{request_id}] [{arch_type}] [PERF] 프로세스 생성: {process_create_elapsed:.4f}초")
            
            # 실시간으로 stdout과 stderr를 읽어서 출력 및 WebSocket 전송
            async def stream_output(stream, stream_name):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    text = line.decode('utf-8', errors='ignore').rstrip()
                    if text:
                        print(f"[{request_id}] [{arch_type}] [{stream_name}] {text}")
                        # WebSocket으로 진행 상황 전송
                        await self._send_websocket_message({
                            "type": "gingiva_generation_progress",
                            "request_id": request_id,
                            "arch_type": arch_type,
                            "message": text,
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
            
            # stdout과 stderr를 동시에 스트리밍
            stream_start = time.perf_counter()
            await asyncio.gather(
                stream_output(process.stdout, "STDOUT"),
                stream_output(process.stderr, "STDERR")
            )
            stream_elapsed = time.perf_counter() - stream_start
            print(f"[{request_id}] [{arch_type}] [PERF] 프로세스 실행 및 스트리밍: {stream_elapsed:.4f}초")
            
            # 프로세스 종료 대기
            wait_start = time.perf_counter()
            returncode = await process.wait()
            wait_elapsed = time.perf_counter() - wait_start
            print(f"[{request_id}] [{arch_type}] [PERF] 프로세스 종료 대기: {wait_elapsed:.4f}초")
            
            print(f"[{request_id}] [{arch_type}] 프로세스 종료 (return code: {returncode})")
            
            # 파일 생성 여부 확인
            file_check_start = time.perf_counter()
            output_file = os.path.join(output_path, f"{arch_type}.stl")
            
            if not os.path.exists(output_file):
                error_msg = f"치은 파일이 생성되지 않음 (return code: {returncode})"
                print(f"[{request_id}] [{arch_type}] 실패: {error_msg}")
                raise Exception(error_msg)
            
            file_size = os.path.getsize(output_file)
            file_check_elapsed = time.perf_counter() - file_check_start
            print(f"[{request_id}] [{arch_type}] [PERF] 파일 확인: {file_check_elapsed:.4f}초")
            print(f"[{request_id}] [{arch_type}] 생성 완료: {output_file} ({file_size:,} bytes)")
            
            if returncode != 0:
                print(f"[{request_id}] [{arch_type}] 경고: 프로세스가 오류 코드 {returncode}로 종료되었지만, 파일은 정상 생성됨")
            
            # === 단일 arch_type 전체 성능 로깅 ===
            arch_total_elapsed = time.perf_counter() - arch_start_time
            print(f"[{request_id}] [{arch_type}] [PERF] === {arch_type} 완료 ===")
            print(f"[{request_id}] [{arch_type}] [PERF] 총 소요 시간: {arch_total_elapsed:.4f}초 ({arch_total_elapsed/60:.2f}분)")
            print(f"[{request_id}] [{arch_type}] [PERF] 파일 크기: {file_size:,} bytes")
            
            return {
                "status": "success",
                "generated_files": [{
                    "arch_type": arch_type,
                    "file_path": output_file,
                    "file_size": file_size
                }],
                "performance": {
                    "total_time": arch_total_elapsed,
                    "process_execution_time": stream_elapsed
                }
            }
            
        except Exception as e:
            print(f"[{request_id}] [{arch_type}] 오류 발생: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "arch_type": arch_type
            }
    
    async def generate_gingiva(
        self,
        input_path: str,
        output_path: str,
        arch_types: List[str],
        request_id: str
    ) -> Dict[str, Any]:
        """
        치은 생성 실행
        
        Args:
            input_path: 치아 입력 파일들이 있는 경로
            output_path: 생성된 치은 파일을 저장할 경로
            arch_types: 생성할 치은 타입 리스트 ["maxilla", "mandibular"]
            request_id: 요청 추적을 위한 고유 ID
            
        Returns:
            Dict: 생성 결과 정보
            {
                "status": "success" | "error",
                "generated_files": [{"arch_type": str, "file_path": str}],
                "error": str (오류 발생 시)
            }
        """
        # === 성능 프로파일링 시작 (순차 처리) ===
        total_start_time = time.perf_counter()
        
        try:
            print(f"[{request_id}] 치은 생성 시작 (순차 처리 모드)")
            print(f"[{request_id}] input_path: {input_path}")
            print(f"[{request_id}] output_path: {output_path}")
            print(f"[{request_id}] arch_types: {arch_types}")
            
            # 출력 디렉토리 생성
            dir_start = time.perf_counter()
            os.makedirs(output_path, exist_ok=True)
            dir_elapsed = time.perf_counter() - dir_start
            print(f"[{request_id}] [PERF] 출력 디렉토리 생성: {dir_elapsed:.4f}초")
            print(f"[{request_id}] 출력 디렉토리 생성 완료")
            
            # WebSocket으로 시작 메시지 전송
            ws_start = time.perf_counter()
            await self._send_websocket_message({
                "type": "gingiva_generation_started",
                "request_id": request_id,
                "message": "치은 생성이 시작되었습니다.",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            ws_elapsed = time.perf_counter() - ws_start
            print(f"[{request_id}] [PERF] WebSocket 시작 메시지 전송: {ws_elapsed:.4f}초")
            print(f"[{request_id}] WebSocket 시작 메시지 전송 완료")
            
            # 치은 생성을 별도 프로세스에서 실행
            print(f"[{request_id}] 별도 프로세스에서 치은 생성 실행 중...")
            import json
            
            setup_start = time.perf_counter()
            script_path = os.path.join(os.path.dirname(__file__), "run_gingiva_generation.py")
            arch_types_json = json.dumps(arch_types)
            
            # subprocess를 비동기로 실행 (unbuffered 모드, UTF-8 인코딩)
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'  # 버퍼링 비활성화
            env['PYTHONIOENCODING'] = 'utf-8'  # UTF-8 인코딩 강제
            
            setup_elapsed = time.perf_counter() - setup_start
            print(f"[{request_id}] [PERF] 프로세스 준비: {setup_elapsed:.4f}초")
            
            process_start = time.perf_counter()
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-u",  # unbuffered 모드
                script_path,
                input_path,
                output_path,
                arch_types_json,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            process_create_elapsed = time.perf_counter() - process_start
            print(f"[{request_id}] [PERF] 프로세스 생성: {process_create_elapsed:.4f}초")
            
            # 실시간으로 stdout과 stderr를 읽어서 출력 및 WebSocket 전송
            async def stream_output(stream, stream_name):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    text = line.decode('utf-8', errors='ignore').rstrip()
                    if text:
                        print(f"[{request_id}] [{stream_name}] {text}")
                        # WebSocket으로 진행 상황 전송
                        await self._send_websocket_message({
                            "type": "gingiva_generation_progress",
                            "request_id": request_id,
                            "message": text,
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
            
            # stdout과 stderr를 동시에 스트리밍
            stream_start = time.perf_counter()
            await asyncio.gather(
                stream_output(process.stdout, "STDOUT"),
                stream_output(process.stderr, "STDERR")
            )
            stream_elapsed = time.perf_counter() - stream_start
            print(f"[{request_id}] [PERF] 프로세스 실행 및 스트리밍: {stream_elapsed:.4f}초")
            
            # 프로세스 종료 대기
            wait_start = time.perf_counter()
            returncode = await process.wait()
            wait_elapsed = time.perf_counter() - wait_start
            print(f"[{request_id}] [PERF] 프로세스 종료 대기: {wait_elapsed:.4f}초")
            
            print(f"[{request_id}] 치은 생성 프로세스 종료 (return code: {returncode})")
            
            # 파일 생성 여부로 성공/실패 판단 (외부 라이브러리의 인코딩 오류 무시)
            file_check_start = time.perf_counter()
            generated_files = []
            for arch_type in arch_types:
                output_file = os.path.join(output_path, f"{arch_type}.stl")
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    print(f"[{request_id}] 생성 확인: {output_file} ({file_size:,} bytes)")
                    generated_files.append({
                        "arch_type": arch_type,
                        "file_path": output_file
                    })
            file_check_elapsed = time.perf_counter() - file_check_start
            print(f"[{request_id}] [PERF] 파일 확인: {file_check_elapsed:.4f}초")
            
            # 파일이 하나도 생성되지 않았으면 실패
            if not generated_files:
                error_msg = f"치은 파일이 생성되지 않음 (return code: {returncode})"
                print(f"[{request_id}] 치은 생성 실패: {error_msg}")
                raise Exception(error_msg)
            
            # 파일이 생성되었으면 성공 (returncode가 1이어도 OK)
            if returncode != 0:
                print(f"[{request_id}] 경고: 프로세스가 오류 코드 {returncode}로 종료되었지만, 파일은 정상 생성됨")
            
            print(f"[{request_id}] 치은 생성 프로세스 완료 (파일 {len(generated_files)}개 생성)")
            print(f"[{request_id}] 치은 생성 완료")
            
            # WebSocket으로 완료 메시지 전송
            ws_complete_start = time.perf_counter()
            await self._send_websocket_message({
                "type": "gingiva_generation_completed",
                "request_id": request_id,
                "generated_files": generated_files,
                "message": "치은 생성이 완료되었습니다.",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            ws_complete_elapsed = time.perf_counter() - ws_complete_start
            print(f"[{request_id}] [PERF] WebSocket 완료 메시지 전송: {ws_complete_elapsed:.4f}초")
            
            # === 전체 성능 로깅 (순차 처리) ===
            total_elapsed = time.perf_counter() - total_start_time
            print(f"[{request_id}] [PERF] ========================================")
            print(f"[{request_id}] [PERF] 전체 실행 시간: {total_elapsed:.4f}초 ({total_elapsed/60:.2f}분)")
            print(f"[{request_id}] [PERF] 처리 모드: 순차 처리")
            print(f"[{request_id}] [PERF] arch_type 개수: {len(arch_types)}")
            print(f"[{request_id}] [PERF] 생성된 파일 수: {len(generated_files)}")
            print(f"[{request_id}] [PERF] ========================================")
            
            return {
                "status": "success",
                "generated_files": generated_files,
                "performance": {
                    "total_time": total_elapsed,
                    "processing_time": stream_elapsed,
                    "arch_types_count": len(arch_types),
                    "files_generated": len(generated_files)
                }
            }
            
        except Exception as e:
            total_elapsed = time.perf_counter() - total_start_time
            print(f"[{request_id}] [PERF] 오류 발생 전까지 실행 시간: {total_elapsed:.4f}초")
            print(f"[{request_id}] 치은 생성 중 오류 발생: {str(e)}")
            
            # WebSocket으로 오류 메시지 전송
            await self._send_websocket_message({
                "type": "gingiva_generation_failed",
                "request_id": request_id,
                "error": str(e),
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _send_websocket_message(self, message: Dict[str, Any]) -> None:
        """
        WebSocket을 통해 메시지 전송
        
        Args:
            message: 전송할 메시지 딕셔너리
        """
        if self.websocket:
            try:
                await self.websocket.send_json(message)
            except Exception as e:
                print(f"WebSocket 메시지 전송 실패: {str(e)}")
    
    def validate_arch_types(self, arch_types: List[str]) -> tuple[bool, Optional[str]]:
        """
        arch_types 유효성 검증
        
        Args:
            arch_types: 검증할 arch_types 리스트
            
        Returns:
            tuple: (유효 여부, 오류 메시지)
        """
        valid_arch_types = ["maxilla", "mandibular"]
        
        if not arch_types:
            return False, "arch_types가 비어있습니다."
        
        for arch_type in arch_types:
            if arch_type not in valid_arch_types:
                return False, f"유효하지 않은 arch_type: {arch_type}. 'maxilla' 또는 'mandibular'만 가능합니다."
        
        return True, None
    
    @staticmethod
    def validate_input_path(input_path: str) -> tuple[bool, Optional[str]]:
        """
        입력 경로 유효성 검증
        
        Args:
            input_path: 검증할 입력 경로
            
        Returns:
            tuple: (유효 여부, 오류 메시지)
        """
        if not input_path:
            return False, "input_path가 비어있습니다."
        
        if not os.path.exists(input_path):
            return False, f"입력 경로가 존재하지 않습니다: {input_path}"
        
        return True, None

