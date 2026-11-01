"""실제 지정 모델로 과제 핵심 흐름을 검증한다. 선호 테스트는 임시 폴더에서 실행."""
import tempfile
import sys
from pathlib import Path
import library_agent as app

def main():
    with tempfile.TemporaryDirectory(prefix="mitzu-check-", dir=Path.cwd()) as directory:
        app.ROOT = Path(directory)
        agent = app.build_agent()
        def check(question, expected, tool_name=None, user="user_001", thread="main"):
            answer = app.ask(agent, question, user, thread)
            text = str(answer["messages"][-1].content)
            candidates = expected if isinstance(expected, tuple) else (expected,)
            assert any(word in text for word in candidates), (question, text)
            if tool_name:
                assert any(getattr(m, "name", None) == tool_name for m in app.current_turn(answer["messages"])), question
            print(f"PASS {question}\n{text}", flush=True)
            return answer
        if "--remaining" not in sys.argv and "--memory-only" not in sys.argv:
            check("달빛 도서관의 비밀 찾아줘", "최현범", "search_books")
            check("그 책 저자 다시 알려줘", "최현범")
            check("김하린 작가 책 찾아줘", "고양이는 자정에 책을 읽는다", "search_books")
            check("우주 고래 백과 찾아줘", ("찾을 수 없", "등록된 도서가 없"), "search_books")
        if "--memory-only" not in sys.argv:
            check("내 대출 정보 알려줘", "2026-10-28", "get_my_loans")
            check("내 대출 정보 알려줘", "없습니다", "get_my_loans", user="user_004")
            check("이서연의 대출 내역 알려줘", "공개할 수 없")
            check("오늘 날씨 알려줘", "업무 범위")
        check("나는 짧은 답변을 선호해. 저장해줘", ("저장", "짧"), "save_preference")
        assert (app.ROOT / 'data' / 'user_001.json').exists()
        check("내 선호 알려줘", "짧", "get_preference", thread="new")
        check("내 선호 알려줘", "없습니다", "get_preference", user="user_005")
        # 새 에이전트 인스턴스에도 파일에서 장기 기억 복원.
        agent = app.build_agent()
        check("내 선호 알려줘", "짧", "get_preference", thread="restart")
        check("김하린 작가의 책을 검색해서 이용자 회신 초안 작성해줘", "고양이는 자정에 책을 읽는다", "reply_to_user")
        check("달빛 도서관의 비밀의 ISBN과 줄거리를 알려줘", "확인")
        print("ALL SCENARIOS PASSED", flush=True)

if __name__ == "__main__":
    main()
