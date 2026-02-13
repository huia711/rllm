import multiprocessing as mp
import traceback

import gymnasium as gym

from rllm.environments.base.base_env import BaseEnv


class BrowserGymEnv(BaseEnv):
    def __init__(self, env_id="browsergym/openended", task=None, **env_kwargs):
        self.parent_conn, self.child_conn = mp.Pipe()
        self.timeout = env_kwargs.pop("timeout", None)  # in seconds
        self.process = mp.Process(target=self._worker, args=(self.child_conn, env_id, task, env_kwargs))
        self.process.start()

    def _worker(self, conn, env_id, task, env_kwargs):
        # Import browsergym modules in the worker process to register environments
        try:
            import browsergym.miniwob  # noqa: F401
        except ImportError:
            pass

        browser_launch_args = [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-application-cache",
            "--disk-cache-size=1",
            "--media-cache-size=1",
            "--disable-cache",
            "--disable-gpu",
            "--disable-software-rasterizer",
            "--incognito",
        ]

        if task:
            task_kwargs = dict(task)
            # Backward compatibility: older rLLM scripts use miniwob_url/subtask.
            task_kwargs.pop("subtask", None)
            if "miniwob_url" in task_kwargs and "base_url" not in task_kwargs:
                task_kwargs["base_url"] = task_kwargs.pop("miniwob_url")
            else:
                task_kwargs.pop("miniwob_url", None)

            make_kwargs = {
                "task_kwargs": task_kwargs,
                **env_kwargs,
            }

            try:
                from browsergym.core.env import BrowserEnv

                browser_env_params = BrowserEnv.__init__.__code__.co_varnames
                if "browser_args" in browser_env_params:
                    make_kwargs["browser_args"] = browser_launch_args
                    make_kwargs["user_data_dir"] = None  # Forces incognito on older versions
                elif "pw_chromium_kwargs" in browser_env_params:
                    # Newer browsergym already sets launch args internally.
                    # Only disable chromium sandbox for containerized environments.
                    make_kwargs["pw_chromium_kwargs"] = {"chromium_sandbox": False}
            except Exception:
                # Fallback to defaults if introspection fails.
                pass

            env = gym.make(env_id, **make_kwargs)
        else:
            env = gym.make(env_id, **env_kwargs)

        try:
            while True:
                cmd, data = conn.recv()
                if cmd == "reset":
                    try:
                        obs = env.reset()
                        conn.send(obs)
                    except Exception as e:
                        conn.send(("__error__", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
                elif cmd == "step":
                    try:
                        action = data
                        obs, reward, terminated, truncated, extra_info = env.step(action)
                        conn.send((obs, reward, terminated or truncated, extra_info))
                    except Exception as e:
                        conn.send(("__error__", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
                elif cmd == "close":
                    env.close()
                    conn.close()
                    break
        except EOFError:
            env.close()

    def reset(self):
        self.parent_conn.send(("reset", None))
        if self.timeout is not None:
            if not self.parent_conn.poll(self.timeout):
                raise TimeoutError(f"Timeout after {self.timeout} seconds waiting for response.")
        try:
            result = self.parent_conn.recv()
        except EOFError as e:
            raise RuntimeError("BrowserGym worker process terminated during reset.") from e
        if isinstance(result, tuple) and len(result) == 2 and result[0] == "__error__":
            raise RuntimeError(f"BrowserGym reset failed in worker:\n{result[1]}")
        return result

    def step(self, action):
        self.parent_conn.send(("step", action))
        if self.timeout is not None:
            if not self.parent_conn.poll(self.timeout):
                raise TimeoutError(f"Timeout after {self.timeout} seconds waiting for response.")
        try:
            result = self.parent_conn.recv()
        except EOFError as e:
            raise RuntimeError("BrowserGym worker process terminated during step.") from e
        if isinstance(result, tuple) and len(result) == 2 and result[0] == "__error__":
            raise RuntimeError(f"BrowserGym step failed in worker:\n{result[1]}")
        return result

    def close(self):
        self.parent_conn.send(("close", None))
        self.process.join(60 * 2)
        if self.process.is_alive():
            print(f"Process still alive after {self.timeout} seconds. Killing it.")
            self.process.terminate()
            self.process.join()

    @staticmethod
    def from_dict(extra_info: dict) -> "BrowserGymEnv":
        # Keep `env_id` as explicit constructor arg and pass remaining fields as either
        # browser env kwargs or task kwargs for miniwob-style tasks.
        info = dict(extra_info)
        env_id = info.pop("env_id")
        headless = info.pop("headless", True)
        timeout_ms = info.pop("timeout", None)

        # Common task fields passed from dataset/env_args.
        task = {}
        for key in ("task_kwargs", "subtask", "miniwob_url", "base_url", "seed"):
            if key in info:
                if key == "task_kwargs" and isinstance(info[key], dict):
                    task.update(info.pop(key))
                else:
                    task[key] = info.pop(key)

        # Backward compatibility conversion for miniwob.
        task.pop("subtask", None)
        if "miniwob_url" in task and "base_url" not in task:
            task["base_url"] = task.pop("miniwob_url")
        else:
            task.pop("miniwob_url", None)

        env_kwargs = {"headless": headless, **info}
        if timeout_ms is not None:
            env_kwargs["timeout"] = timeout_ms / 1000.0 if timeout_ms > 10 else timeout_ms

        return BrowserGymEnv(env_id=env_id, task=(task or None), **env_kwargs)

    @staticmethod
    def is_multithread_safe() -> bool:
        return True
