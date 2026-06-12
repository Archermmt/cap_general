success = libero_vla_episode(task='open the middle drawer of the cabinet', max_steps=300)
print(f"Episode result: {'SUCCESS' if success else 'FAIL'}")
RESULT = {"success": success, "task": 'open the middle drawer of the cabinet'}
