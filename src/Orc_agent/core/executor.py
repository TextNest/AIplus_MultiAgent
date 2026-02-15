
import sys
import io
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import koreanize_matplotlib

class PersistentPythonExecutor:
    def __init__(self):
        import matplotlib.font_manager as fm
    
        font_list = ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'NanumBarunGothic']
        self.available_font = None
    
        for font in font_list:
            if any(font in f.name for f in fm.fontManager.ttflist):
                self.available_font = font
                break
        else:
            self.available_font = None

        if self.available_font:
            plt.rcParams['font.family'] = self.available_font
            plt.rcParams['axes.unicode_minus'] = False
            print(f"한글 폰트 설정: {self.available_font}")
        else:
            print("경고: 한글 폰트를 찾을 수 없습니다")
        
        self.globals = {
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "os": os,
            "io": io,
        }
        self.globals["__builtins__"] = __builtins__

    def run(self, code: str) -> str:
        """
        코드를 실행하고 표준 출력을 캡처하여 반환합니다.
        에러 발생 시 Traceback을 반환합니다.
        """
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        try:
            # 코드 실행 (지속되는 globals 사용)
            exec(code, self.globals)
            
            result = redirected_output.getvalue()
            return result.strip() if result else "Success"
            
        except Exception:
            # 에러 발생 시 Traceback 캡처
            return traceback.format_exc()
        finally:
            sys.stdout = old_stdout

    def get_globals_keys(self):
        return list(self.globals.keys())

# 싱글톤 인스턴스 생성
executor_instance = PersistentPythonExecutor()
