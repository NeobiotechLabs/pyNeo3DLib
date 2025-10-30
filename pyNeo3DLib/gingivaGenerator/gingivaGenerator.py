import os
import datetime
from typing import List, Dict, Any, Optional
from single_template_maker_lib import TeethTemplateMaker


class GingivaGenerator:
    """
    치은(Gingiva) 생성 클래스
    
    TeethTemplateMaker를 사용하여 치아 입력 데이터로부터 치은 메쉬를 생성합니다.
    """
    
    def __init__(self, websocket=None):
        """
        GingivaGenerator 초기화
        
        Args:
            websocket: WebSocket 연결 객체 (선택적, 진행 상황 알림용)
        """
        self.websocket = websocket
    
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
            arch_types: 생성할 치은 타입 리스트 ["maxillary", "mandibular"]
            request_id: 요청 추적을 위한 고유 ID
            
        Returns:
            Dict: 생성 결과 정보
            {
                "status": "success" | "error",
                "generated_files": [{"arch_type": str, "file_path": str}],
                "error": str (오류 발생 시)
            }
        """
        try:
            print(f"[{request_id}] 치은 생성 시작")
            
            # 출력 디렉토리 생성
            os.makedirs(output_path, exist_ok=True)
            
            # WebSocket으로 시작 메시지 전송
            await self._send_websocket_message({
                "type": "gingiva_generation_started",
                "request_id": request_id,
                "message": "치은 생성이 시작되었습니다.",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # 치은 생성 실행
            template_maker = TeethTemplateMaker(input_path, output_path, arch_types)
            template_maker.run()
            
            print(f"[{request_id}] 치은 생성 완료")
            
            # 생성된 파일 목록
            generated_files = []
            for arch_type in arch_types:
                output_file = os.path.join(output_path, f"{arch_type}.stl")
                if os.path.exists(output_file):
                    generated_files.append({
                        "arch_type": arch_type,
                        "file_path": output_file
                    })
            
            # WebSocket으로 완료 메시지 전송
            await self._send_websocket_message({
                "type": "gingiva_generation_completed",
                "request_id": request_id,
                "generated_files": generated_files,
                "message": "치은 생성이 완료되었습니다.",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            return {
                "status": "success",
                "generated_files": generated_files
            }
            
        except Exception as e:
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
        valid_arch_types = ["maxillary", "mandibular"]
        
        if not arch_types:
            return False, "arch_types가 비어있습니다."
        
        for arch_type in arch_types:
            if arch_type not in valid_arch_types:
                return False, f"유효하지 않은 arch_type: {arch_type}. 'maxillary' 또는 'mandibular'만 가능합니다."
        
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

