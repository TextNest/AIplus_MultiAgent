import os

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from .observe import create_callback_handler, is_langfuse_enabled
from .config import config

class LLMFactory:
    @staticmethod
    def create(
        provider: str,
        model: str,
        temperature: float = 0,
    ):
        """
        LLM 객체와 Langfuse Callbacks를 세트로 반환합니다.
        
        Args:
            provider: 'google', 'openai', 'anthropic' 중 하나
            model: 모델 이름 (예: 'gemma-3-27b-it', 'gpt-4o', 'claude-3-5-sonnet')
            temperature: 생성 온도 (기본값: 0)
        
        Returns:
            tuple: (llm, callbacks)
        
        사용 예시:
            from src.core.llm_factory import LLMFactory
            from src.core.observe import langfuse_session
            
            llm, callbacks = LLMFactory.create('google', 'gemma-3-27b-it')
            
            # session_id를 기록하려면 langfuse_session 컨텍스트 사용
            with langfuse_session(session_id="my-session-id") as lf_metadata:
                response = llm.invoke(prompt, config={
                    'callbacks': callbacks,
                    'metadata': lf_metadata,
                })
        """
        # 1. 모델 객체 생성
        if provider == "google":
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=os.environ.get("GOOGLE_API_KEY"),
                temperature=temperature,
            )
        elif provider == "openai":
            llm = ChatOpenAI(
                model=model,
                api_key=os.environ.get("OPENAI_API_KEY"),
                temperature=temperature,
            )
        elif provider == "anthropic":
            llm = ChatAnthropic(
                model=model,
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                temperature=temperature,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # 2. Langfuse Callback 생성 (SessionAwareCallbackHandler 사용)
        callbacks = []
        
        handler = create_callback_handler()
        if handler is not None:
            callbacks.append(handler)
            
        return llm, callbacks
    
# DummyLLM 클래스 추가 (LLMFactory 클래스 위쪽)
class DummyLLM:
    def invoke(self, prompt, config=None):
        prompt_str = str(prompt)
        print(f">>> [DEBUG] Check Prompt: {prompt_str[:50]}...")

        class DummyResponse:
            def __init__(self, code=None, content=None, overall_insight=None, image_specific_insights=None):
                self.code = code
                self.content = content
                self.overall_insight = overall_insight
                self.image_specific_insights = image_specific_insights

        # 리포트 Supervisor
        if "supervisor" in prompt_str.lower() or "next_worker" in prompt_str.lower():
            pass

        # 리포트 내용 생성 (generate_content)
        elif "report" in prompt_str:
            return DummyResponse(content="# TEST\n\n이것은 더미 데이터 분석 보고서입니다.\n\n## 분석 결과\n- 결과 1: 좋음\n- 결과 2: 나쁨")


        # 1. 평가 (Eval) - 가장 먼저 체크!
        # 키워드: '검증 전문가', 'LLM-as-a-judge'
        if "검증 전문가" in prompt_str or "LLM-as-a-judge" in prompt_str:
            return DummyResponse(content="APPROVE")
        # 2. 코드 생성 (Make)
        # 키워드: '파이썬 코드', 'MakeCodeOutput'
        elif "MakeCodeOutput" in prompt_str or "파이썬 코드" in prompt_str:
            dirty_code = (
                "import matplotlib\n"
                "matplotlib.use('Agg')\n"
                "import pandas as pd\n"
                "import matplotlib.pyplot as plt\n"
                "import os\n"
                "current_dir = os.getcwd().replace('\\\\', '/')\n"
                "img_dir = f'{current_dir}/img'\n"
                "if not os.path.exists(img_dir): os.makedirs(img_dir)\n"
                "plt.figure()\n"
                "plt.plot([1, 2, 3], [1, 4, 9])\n"
                "plt.title('Dummy Graph')\n"
                "save_path = f'{img_dir}/figure_0_0.png'\n"
                "plt.savefig(save_path)\n"
                "plt.close()\n"
                "print(f'Saved: {save_path}')"
            )
            return DummyResponse(code=dirty_code, content="코드 생성 완료")
        # 3. 인사이트 (Insight)
        # 키워드: '수석 데이터 분석가', '종합 인사이트'
        elif "수석 데이터 분석가" in prompt_str or "종합 인사이트" in prompt_str:
            return DummyResponse(
                overall_insight="데이터 전반적으로 상승 추세입니다.",
                image_specific_insights=[]
            )
        # 4. 분석 계획 (Plan) - 나머지
        # 키워드: '분석 계획', '마케팅 데이터 전략가'
        elif "분석 계획" in prompt_str or "마케팅 데이터 전략가" in prompt_str:
            return DummyResponse(content="1. 데이터 로드\n2. 컬럼별 기초 통계 확인\n3. 시각화 수행")

        return DummyResponse(content="Dummy Response")
    def with_structured_output(self, schema):
        return self
    
# LLMFactory.create 메서드 수정
class LLMFactory:
    @staticmethod
    def create(provider: str, model: str, temperature: float = 0):
        # [추가] 더미 모드 체크
        if config.DUMMY_MODE:
            return DummyLLM(), None    