"""인증 성공·실패, 로그인 우회 방지와 로그아웃 계정 전환 검증. 모델 호출 없음."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import auth
import library_agent as app

def main():
    with tempfile.TemporaryDirectory(prefix="mitzu-login-", dir=Path.cwd()) as directory:
        root = Path(directory)
        accounts = auth.ensure_accounts(root)
        for i, uid in enumerate(auth.ACCOUNT_IDS, 1):
            assert auth.authenticate(accounts, uid, f"Mitzu{1000+i}!")
            assert not auth.authenticate(accounts, uid, "wrong")
        assert not auth.authenticate(accounts, "../../unknown", "password")
        assert "Mitzu1001!" not in (root / "data/accounts.json").read_text()
        assert auth.ensure_accounts(root) == accounts
        with patch('builtins.input', return_value='user_001'), patch('getpass.getpass', return_value='wrong'):
            assert auth.login(root) is None
        with patch.object(app, 'ROOT', root), patch('sys.argv', ['library_agent.py', '--user', 'user_001', '--ask', '내 대출']), patch('builtins.input', return_value=''), patch('getpass.getpass', return_value='wrong'), patch.object(app, 'build_agent') as build:
            app.main()
            build.assert_not_called()
        seen = []
        def fake_ask(agent, text, user_id, thread):
            seen.append((user_id, thread))
            return {'messages': [SimpleNamespace(content='확인')]}
        inputs = ['user_001', '내 대출', '/logout', 'user_002', '내 대출', '/quit']
        with patch.object(app, 'ROOT', root), patch('sys.argv', ['library_agent.py']), patch('builtins.input', side_effect=inputs), patch('getpass.getpass', side_effect=['Mitzu1001!', 'Mitzu1002!']), patch.object(app, 'build_agent', return_value=object()), patch.object(app, 'ask', side_effect=fake_ask):
            app.main()
        assert [uid for uid, thread in seen] == ['user_001', 'user_002']
        assert seen[0][1] != seen[1][1]
        print('PASS: 5 accounts, wrong password, hash storage, login bypass prevention, logout identity isolation')

if __name__ == '__main__':
    main()
