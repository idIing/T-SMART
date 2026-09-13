import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsqa.eval.provenance import run_provenance


class TestProvenance(unittest.TestCase):
    def test_has_required_keys_and_is_json_serializable(self):
        prov = run_provenance()
        for key in (
            "timestamp_utc", "git_commit", "git_branch", "git_dirty",
            "argv", "cwd", "python", "platform", "hostname",
            "packages", "tsqa_package_path",
        ):
            self.assertIn(key, prov)
        # must round-trip through JSON (no non-serializable objects)
        json.loads(json.dumps(prov, default=str))
        self.assertIsInstance(prov["packages"], dict)
        self.assertIsInstance(prov["git_dirty"], bool)

    def test_extra_is_merged(self):
        prov = run_provenance(extra={"dataset": {"n_rows": 7}})
        self.assertEqual(prov["dataset"]["n_rows"], 7)

    def test_does_not_leak_machine_identity(self):
        """These manifests are committed and published: no absolute paths, no hostname.

        Regression guard for the leaked ``/home/<user>/...`` cwd and machine name
        that shipped in the first round of committed run logs.
        """
        prov = run_provenance()
        for key in ("cwd", "tsqa_package_path"):
            value = prov[key]
            if value is not None:
                self.assertFalse(
                    os.path.isabs(value), f"{key} must be repo-relative, got {value!r}"
                )
        self.assertIsNone(prov["hostname"], "hostname must be opt-in via TSQA_PROVENANCE_HOSTNAME")

    def test_hostname_is_opt_in(self):
        os.environ["TSQA_PROVENANCE_HOSTNAME"] = "1"
        try:
            self.assertIsNotNone(run_provenance()["hostname"])
        finally:
            del os.environ["TSQA_PROVENANCE_HOSTNAME"]

    def test_never_raises_without_git(self):
        # Running from a non-repo dir must still return a dict (git facts None).
        cwd = os.getcwd()
        try:
            os.chdir(os.path.sep)  # filesystem root, not a git repo
            prov = run_provenance()
            self.assertIsInstance(prov, dict)
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
