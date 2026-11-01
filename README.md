# 미쯔 AI 사서 보조 에이전트

LangChain 수업의 `05_full_agent.py` 통합 구조를 도서관 업무에 적용한 로컬 싱글 에이전트입니다.
사서 이름은 **미쯔**, 역할은 **AI 사서 보조**이며 Ollama의
`qwen3:4b-instruct-2507-q4_K_M` 모델을 사용합니다.

## 주요 기능

- 과제로 제공된 도서 11권을 제목·저자로 검색
- 로그인 이용자 본인의 가상 대출 기록·반납 예정일 조회
- 도구로 확인한 내용에 기반한 이용자 회신 초안
- Runtime Context에 사서 이름·역할·로그인 이용자 ID 전달
- 개인정보·규칙 우회·업무 외 요청 등의 입력 가드레일
- 대화별 단기 기억과 이용자별 선호 장기 기억
- Judge 및 미제공 정보 규칙을 통한 최종 답변 검수
- 추가 기능: 출처와 기간을 표시하는 저장형 YES24 베스트셀러 조회

외부 도서관 API, 별도 RAG, 유료 API 키는 사용하지 않습니다.
베스트셀러는 모델 학습이나 실시간 조회가 아니라 로컬 JSON 자료 조회입니다.

## 설치와 실행

Python 3.12 및 Ollama를 준비합니다. 최초 실행 전에 지정 모델을 받으세요.

```powershell
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

이 저장소를 내려받은 폴더에서 실행합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run.ps1
```

PowerShell 실행 정책으로 ps1 실행이 제한되면 Python을 직접 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe library_agent.py
```

macOS/Linux에서도 Python 가상환경에 requirements를 설치한 뒤 `python library_agent.py`로 실행합니다.
Ollama 서버는 `http://127.0.0.1:11434`에서 실행되어야 합니다.

## 가상 이용자 로그인

| ID | 가상 이름 | 실습용 초기 비밀번호 |
|---|---|---|
| user_001 | 김민준 | Mitzu1001! |
| user_002 | 이서연 | Mitzu1002! |
| user_003 | 박지호 | Mitzu1003! |
| user_004 | 최하은 | Mitzu1004! |
| user_005 | 정도윤 | Mitzu1005! |

비밀번호 입력은 화면에 표시되지 않습니다. 초기 계정은 로컬 수업용이며 실제 서비스 인증이 아닙니다.
최초 실행 시 계정별 salt와 PBKDF2-SHA256 해시가 `data/accounts.json`에 생성됩니다.
비밀번호·계정 해시는 모델에 전달하지 않습니다. 3회 실패하면 종료됩니다.
`--user` 또는 `-UserId`는 ID 기본값이며 비밀번호 인증을 생략하지 않습니다.

| 명령 | 동작 |
|---|---|
| `/new` | 새 대화 시작, 장기 선호는 유지 |
| `/logout` | 로그아웃 후 다른 계정으로 로그인 |
| `/quit` | 종료 |

## 질문 예시

```text
달빛 도서관의 비밀 찾아줘
그 책 저자 다시 알려줘
김하린 작가 책 찾아줘
내 대출 정보 알려줘
이서연 대출 내역 알려줘
오늘 날씨 알려줘
짧은 답변을 선호해. 저장해줘
내 선호 알려줘
김하린 작가의 책을 찾아서 이용자 회신 초안 작성해줘
베스트셀러 상위 3권 알려줘
```

선호를 저장하고 `/new`를 입력한 다음 `내 선호 알려줘`로 다른 대화에서의 기억 조회를 확인합니다.
user_004는 가상 대출 기록이 없습니다.

자동 데모도 로그인 후 실행됩니다.

```powershell
.\run.ps1 -Demo
```

## 수업 기술과 코드 대응

| 수업 기술 | 코드 | 역할 |
|---|---|---|
| 도구 | `@tool` 함수 6개 | 도서·대출·선호·회신·베스트셀러 조회 |
| Runtime | `Context` | 로그인 신원과 사서 설정 |
| 입력 훅 | `library_guardrail` | 카테고리별 조기 종료 |
| 호출 개입 | `enrich_context` | 응대 원칙과 필요한 도구만 노출 |
| 단기 기억 | `InMemorySaver`, `thread_id` | 대화 기록 유지 |
| 장기 기억 | `InMemoryStore`, `ToolRuntime` | 사용자 namespace별 선호 저장·조회 |
| 최종 검수 | `final_answer_review` | Judge 판정·규칙 검사·동일 메시지 ID 교체 |

`service_policy/facts`는 매 호출 주입하고 `preference`는 필요할 때 도구로 조회합니다.
선호는 고정 key를 덮어쓰는 Upsert 방식이며 이용자별 JSON으로 저장·복원합니다.
InMemoryStore 자체는 영구 저장소가 아닙니다.
작은 모델의 도구 호출 누락을 보완하기 위해 명시적인 선호 저장과 회신 작성에는 보완 실행 경로가 있습니다.

## 파일 구성

```text
library_agent.py       에이전트·도구·미들웨어·CLI
auth.py                실습용 인증
bestsellers.json       출처·기간이 있는 저장형 순위
run.ps1                프로젝트 가상환경 실행
requirements.txt       사용한 주요 패키지 버전
verify_login.py        인증·로그아웃 분리 검증
verify_course_flow.py  입력 가드레일 검증
verify_scenarios.py    실제 모델 시나리오 검증
```

## 검증

```powershell
.\.venv\Scripts\python.exe verify_login.py
.\.venv\Scripts\python.exe verify_course_flow.py
.\.venv\Scripts\python.exe verify_scenarios.py
```

앞의 두 검증은 로그인 흐름과 입력 차단을 확인하며 실제 생성 모델의 성능 평가와 다릅니다.
마지막 검증은 Ollama 모델을 호출합니다. 로컬 모델의 표현과 도구 선택이 달라 일부 검사가 실패할 수 있습니다.
실행 시간이 오래 걸릴 수 있으며 정확도 100%나 환각 완전 제거를 보장하지 않습니다.

## 베스트셀러 출처와 범위

YES24 국내도서 종합 베스트 상위 10권을 2026-10-19에 확인한 참고 데이터입니다.
집계 기간은 **2026-10-12 ~ 2026-10-18**이며 현재 대한민국 전체 서점 통합 순위가 아닙니다.

출처: https://www.yes24.com/Product/Category/BestSeller?categoryNumber=001&pageNumber=1&pageSize=24

소장 목록 11권과 따로 관리하며, 소장 여부는 목록과 대조합니다. 자동 갱신되지 않습니다.

## 한계

- 도서·대출·계정은 수업용 가상 데이터입니다.
- 실제 대출·예약·반납·메시지 발송은 수행하지 않습니다.
- 도서의 장르·줄거리·ISBN·출판사·실제 재고·도서관 운영시간은 확인할 데이터가 없습니다.
- 키워드 기반 가드레일과 Judge에는 우회·오탐 가능성이 있습니다.
- 기본 비밀번호가 공개된 로컬 실습이며 서비스용 인증으로 사용하면 안 됩니다.
- `data/`, `.env`, 가상환경, 실제 계정 해시·이용자 선호 데이터는 업로드하지 않습니다.

논문 PDF와 데모 영상은 이 코드 저장소에 아직 포함되어 있지 않습니다.
