"""matplotlib 한글 폰트 자동 설정.

사용법:
    from utils.korean_font import setup_korean_font
    setup_korean_font()

setup.sh 가 fonts-nanum, fonts-nanum-coding 을 미리 설치해두므로
대부분의 환경에서 자동 동작합니다. 만약 폰트 캐시가 stale 이면:
    rm -rf ~/.cache/matplotlib
    python -c "import matplotlib.font_manager as fm; fm._load_fontmanager(try_read_cache=False)"
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 우선순위 순 (Linux NanumGothic → Noto CJK → macOS AppleGothic → Windows Malgun)
_CANDIDATES = (
    "NanumGothic",
    "NanumBarunGothic",
    "NanumSquare",
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "AppleGothic",
    "Apple SD Gothic Neo",
    "Malgun Gothic",
)


def setup_korean_font(verbose: bool = True) -> str | None:
    """matplotlib 의 기본 폰트를 한글 가능 폰트로 변경. 사용 가능한 첫 후보를 사용.

    Returns:
        설정한 폰트 이름. 후보 폰트가 하나도 없으면 None.
    """
    available = {f.name for f in fm.fontManager.ttflist}
    for cand in _CANDIDATES:
        if cand in available:
            plt.rcParams["font.family"] = cand
            plt.rcParams["axes.unicode_minus"] = False   # 마이너스 부호 깨짐 방지
            if verbose:
                print(f"✅ matplotlib 한글 폰트: {cand}")
            return cand

    if verbose:
        print("⚠ 한글 폰트 없음 — 다음 중 한 가지 실행 후 재시도:")
        print("   sudo apt install fonts-nanum fonts-nanum-coding   # Linux")
        print("   rm -rf ~/.cache/matplotlib                         # cache 재구축")
    return None
