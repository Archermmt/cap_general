success = libero_vla_episode(task='put the bowl on the stove', max_steps=300)
print(f"Episode result: {'SUCCESS' if success else 'FAIL'}")
RESULT = {"success": success, "task": 'put the bowl on the stove'}
