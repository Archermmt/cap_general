success = libero_vla_episode(task='push the plate to the front of the stove', max_steps=300)
print(f"Episode result: {'SUCCESS' if success else 'FAIL'}")
RESULT = {"success": success, "task": 'push the plate to the front of the stove'}
