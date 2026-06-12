success = libero_vla_episode(task='turn on the stove', max_steps=300)
print(f"Episode result: {'SUCCESS' if success else 'FAIL'}")
RESULT = {"success": success, "task": 'turn on the stove'}
