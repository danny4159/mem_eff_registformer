# Registformer (with hugging face)

이 프로젝트는 **Hugging Face**를 기반으로 Registration을 목적으로 한 의료 영상 합성 네트워크 구현입니다.

현재는 **stage1: padain_synthesis**까지 개발이 완료되었으며, 차후 registformer(등록 네트워크)도 추가될 예정입니다.

---

## 데모 실행 방법

1. **데이터 및 가중치 다운로드**  
   아래 구글 드라이브에서 `data`와 `weights` 폴더를 다운로드하세요.  
   [데이터 및 가중치 다운로드](https://drive.google.com/drive/folders/1TTkzE9woRUs8ncvi_l5Y-rRP7EmpdaEF?usp=sharing)

2. **프로젝트 루트에 폴더 배치**  
   다운로드한 `data`와 `weights` 폴더 내의 데이터를 프로젝트 최상위(root)의 해당 디렉토리에 넣어주세요.

3. **튜토리얼 실행**  
   Jupyter Notebook에서 아래 튜토리얼을 실행하면 됩니다.
   ```
   nbs/padain_synthesis_tutorial.ipynb
   ```


train.py와 test.py도 실행 가능.


## 환경 세팅 (uv 사용)

프로젝트에 필요한 Python 라이브러리 설치는 uv를 활용하면 빠르고 편리하게 할 수 있습니다.

```bash
uv venv --python=python3.10
source .venv/bin/activate
uv pip install
```

위 명령어를 차례대로 실행하면 pyproject.toml과 uv.lock에 명시된 라이브러리들이 빠르게 설치되어 환경 구성이 완료됩니다.

단, PyTorch는 CUDA 버전에 맞는 별도의 휠을 직접 설치해주셔야 합니다.