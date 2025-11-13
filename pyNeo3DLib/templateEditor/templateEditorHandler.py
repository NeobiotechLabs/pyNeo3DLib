"""
Template Editor Handler

템플릿 편집 세션을 관리하고 teeth-template-editor와 통합합니다.
"""

import datetime
from typing import Dict, Any, Optional, Tuple

# Lazy import: teeth-template-editor는 실제 사용 시점에만 import
# from teeth_template_editor import create_editing_session

class TemplateEditorHandler:
    """
    템플릿 편집 핸들러 클래스
    
    템플릿 편집 세션의 생명주기를 관리하고, 편집 작업의 상태를 추적합니다.
    """
    
    def __init__(self):
        """TemplateEditorHandler 초기화"""
        self.editor_handler = None
        self.config = {
            "blend_template_path": None,
            "stl_export_path": None
        }
        self.session = {
            "is_started": False,
            "blend_template": None,
        }
        self._editor_available = False
        
        # teeth-template-editor 가용성 확인
        try:
            from teeth_template_editor import create_editing_session
            self._create_editing_session = create_editing_session
            self._editor_available = True
        except ImportError:
            self._editor_available = False
            print("Warning: teeth-template-editor not installed.")
    
    def is_editor_available(self) -> bool:
        """템플릿 편집기가 사용 가능한지 확인"""
        return self._editor_available
    
    def is_session_started(self) -> bool:
        """편집 세션이 시작되었는지 확인"""
        return self.session["is_started"]
    
    def _validate_required_fields(self, data: Dict[str, Any], required_fields: list) -> Tuple[bool, Optional[str]]:
        """
        필수 필드 검증
        
        Args:
            data: 검증할 데이터
            required_fields: 필수 필드 리스트
            
        Returns:
            (is_valid, error_message): 검증 결과와 에러 메시지 (있는 경우)
        """
        for field in required_fields:
            if field not in data:
                return False, f"필수 파라미터가 누락되었습니다: {field}"
        return True, None
    
    def _create_response(self, status: str, message: str, request_id: str, 
                        result: Any = None, **kwargs) -> Dict[str, Any]:
        """
        표준화된 응답 생성
        
        Args:
            status: "success" | "error" | "processing"
            message: 응답 메시지
            request_id: 요청 ID
            result: 결과 데이터 (선택적)
            **kwargs: 추가 필드
            
        Returns:
            Dict: 표준화된 응답 딕셔너리
        """
        response = {
            "status": status,
            "message": message,
            "request_id": request_id,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if result is not None:
            response["result"] = result
        
        response.update(kwargs)
        return response
    
    def start_editing(self, request: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """
        템플릿 편집 세션 시작
        
        Args:
            request: 요청 데이터
                {
                    "blend_template_path": str,
                    "stl_export_path": str,
                    "blend_template": str,
                    "arch_degree": float,
                    "y_scale": float
                }
            request_id: 요청 ID
            
        Returns:
            Dict: 응답 데이터
        """
        print(f"[{request_id}] 템플릿 편집 시작 API 호출됨")
        
        # 모듈 가용성 확인
        if not self._editor_available:
            return self._create_response(
                "error",
                "teeth-template-editor 모듈이 설치되지 않았습니다.",
                request_id
            )
        
        # 이미 편집 세션이 시작된 경우
        if self.session["is_started"]:
            return self._create_response(
                "error",
                "편집 세션이 이미 시작되었습니다. 먼저 stop_editing을 호출하세요.",
                request_id
            )
        
        try:
            # 필수 파라미터 검증
            required_fields = ["blend_template_path", "stl_export_path", "blend_template", "arch_degree", "y_scale"]
            is_valid, error_message = self._validate_required_fields(request, required_fields)
            if not is_valid:
                return self._create_response("error", error_message, request_id)
            
            blend_template_path = request["blend_template_path"]
            stl_export_path = request["stl_export_path"]
            blend_template = request["blend_template"]
            arch_degree = request["arch_degree"]
            y_scale = request["y_scale"]
            
            print(f"[{request_id}] 템플릿 경로: {blend_template_path}")
            print(f"[{request_id}] 내보내기 경로: {stl_export_path}")
            print(f"[{request_id}] 템플릿 파일: {blend_template}")
            
            # Handler 초기화 (최초 시작 또는 경로가 변경된 경우)
            if (self.editor_handler is None or 
                self.config["blend_template_path"] != blend_template_path or
                self.config["stl_export_path"] != stl_export_path):
                
                print(f"[{request_id}] 새 편집 세션 생성 중...")
                self.editor_handler = self._create_editing_session(
                    blend_template_path=blend_template_path,
                    stl_export_path=stl_export_path
                )
                self.config["blend_template_path"] = blend_template_path
                self.config["stl_export_path"] = stl_export_path
                print(f"[{request_id}] 편집 세션 생성 완료")
            
            # 편집 시작
            print(f"[{request_id}] 편집 시작...")
            start_result = self.editor_handler.start_editing(
                blend_template=blend_template,
                arch_degree=arch_degree,
                y_scale=y_scale
            )
            
            if start_result is None:
                return self._create_response(
                    "error",
                    "편집 시작 실패",
                    request_id
                )
            
            # 세션 상태 업데이트
            self.session["is_started"] = True
            self.session["blend_template"] = blend_template
            
            print(f"[{request_id}] 편집 시작 완료")
            
            return self._create_response(
                "success",
                "템플릿 편집이 시작되었습니다.",
                request_id,
                result=start_result
            )
            
        except FileNotFoundError as e:
            print(f"[{request_id}] Blender를 찾을 수 없음: {str(e)}")
            return self._create_response(
                "error",
                f"Blender를 찾을 수 없습니다: {str(e)}",
                request_id
            )
        except Exception as e:
            print(f"[{request_id}] 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_response(
                "error",
                f"템플릿 편집 시작 중 오류가 발생했습니다: {str(e)}",
                request_id
            )
    
    def transform(self, request: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """
        템플릿 변환 적용 (여러 번 호출 가능)
        
        Args:
            request: 요청 데이터
                {
                    "arch_degree": float,
                    "y_scale": float
                }
            request_id: 요청 ID
            
        Returns:
            Dict: 응답 데이터
        """
        print(f"[{request_id}] 템플릿 변환 API 호출됨")
        
        # 모듈 가용성 확인
        if not self._editor_available:
            return self._create_response(
                "error",
                "teeth-template-editor 모듈이 설치되지 않았습니다.",
                request_id
            )
        
        # 편집 세션이 시작되지 않은 경우
        if not self.session["is_started"]:
            return self._create_response(
                "error",
                "편집 세션이 시작되지 않았습니다. 먼저 start_editing을 호출하세요.",
                request_id
            )
        
        try:
            # 필수 파라미터 검증
            required_fields = ["arch_degree", "y_scale"]
            is_valid, error_message = self._validate_required_fields(request, required_fields)
            if not is_valid:
                return self._create_response("error", error_message, request_id)
            
            arch_degree = request["arch_degree"]
            y_scale = request["y_scale"]
            
            print(f"[{request_id}] 변환 적용 중... (arch_degree: {arch_degree}, y_scale: {y_scale})")
            transform_result = self.editor_handler.transform(
                arch_degree=arch_degree,
                y_scale=y_scale
            )
            
            if transform_result is None:
                return self._create_response(
                    "error",
                    "변환 적용 실패",
                    request_id
                )
            
            print(f"[{request_id}] 변환 적용 완료")
            
            return self._create_response(
                "success",
                "변환이 성공적으로 적용되었습니다.",
                request_id,
                result=transform_result
            )
            
        except Exception as e:
            print(f"[{request_id}] 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_response(
                "error",
                f"변환 적용 중 오류가 발생했습니다: {str(e)}",
                request_id
            )
    
    def stop_editing(self, request: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """
        템플릿 편집 세션 종료 및 STL 내보내기
        
        Args:
            request: 요청 데이터
                {
                    "arch_degree": float,
                    "y_scale": float
                }
            request_id: 요청 ID
            
        Returns:
            Dict: 응답 데이터
        """
        print(f"[{request_id}] 템플릿 편집 종료 API 호출됨")
        
        # 모듈 가용성 확인
        if not self._editor_available:
            return self._create_response(
                "error",
                "teeth-template-editor 모듈이 설치되지 않았습니다.",
                request_id
            )
        
        # 편집 세션이 시작되지 않은 경우
        if not self.session["is_started"]:
            return self._create_response(
                "error",
                "편집 세션이 시작되지 않았습니다. 먼저 start_editing을 호출하세요.",
                request_id
            )
        
        try:
            # 필수 파라미터 검증
            required_fields = ["arch_degree", "y_scale"]
            is_valid, error_message = self._validate_required_fields(request, required_fields)
            if not is_valid:
                return self._create_response("error", error_message, request_id)
            
            arch_degree = request["arch_degree"]
            y_scale = request["y_scale"]
            
            print(f"[{request_id}] 편집 종료 및 STL 내보내기 중...")
            stop_result = self.editor_handler.stop_editing(
                arch_degree=arch_degree,
                y_scale=y_scale
            )
            
            if stop_result is None:
                return self._create_response(
                    "error",
                    "편집 종료 실패",
                    request_id
                )
            
            # 세션 상태 초기화
            self.session["is_started"] = False
            self.session["blend_template"] = None
            
            print(f"[{request_id}] 템플릿 편집 종료 완료")
            
            return self._create_response(
                "success",
                "템플릿 편집이 완료되고 STL 파일이 내보내졌습니다.",
                request_id,
                result=stop_result,
                stl_folder_path=stop_result.get("stl_folder_path") if stop_result else None
            )
            
        except Exception as e:
            print(f"[{request_id}] 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            # 에러 발생 시에도 세션 상태 초기화 (안전장치)
            self.session["is_started"] = False
            self.session["blend_template"] = None
            return self._create_response(
                "error",
                f"템플릿 편집 종료 중 오류가 발생했습니다: {str(e)}",
                request_id
            )

