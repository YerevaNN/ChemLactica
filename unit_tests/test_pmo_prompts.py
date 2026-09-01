import unittest

from chemlactica.mol_opt.pmo_prompts import (
    PMO_TASKS,
    VALSARTAN_SMARTS_PATTERN,
    get_pmo_additional_properties,
    render_pmo_prefix,
)


class TestPmoPrompts(unittest.TestCase):
    def test_registry_covers_all_pmo_tasks(self):
        self.assertEqual(len(PMO_TASKS), 23)
        for task_name in PMO_TASKS:
            properties = get_pmo_additional_properties(task_name, "task-informed")
            self.assertIsNotNone(properties)

    def test_task_agnostic_mode_has_no_task_derived_prefix(self):
        for task_name in PMO_TASKS:
            self.assertEqual(render_pmo_prefix(task_name, "task-agnostic"), "")

    def test_unsupported_classifier_names_are_not_prompted(self):
        for task_name in ("drd2", "gsk3b", "jnk3"):
            self.assertEqual(render_pmo_prefix(task_name, "task-informed"), "")

    def test_representative_trained_tag_prefixes(self):
        self.assertEqual(render_pmo_prefix("qed", "task-informed"), "[QED]0.95[/QED]")
        self.assertEqual(
            render_pmo_prefix("isomers_c7h8n2o2", "task-informed"),
            "[FORMULA]C7H8N2O2[/FORMULA]",
        )
        self.assertEqual(
            render_pmo_prefix("sitagliptin_mpo", "task-informed"),
            "[CLOGP]2.02[/CLOGP][TPSA]77.04[/TPSA]" "[FORMULA]C16H15F6N5O[/FORMULA]",
        )

    def test_valsartan_uses_exact_smarts_literal_as_similarity_reference(self):
        self.assertEqual(
            render_pmo_prefix("valsartan_smarts", "task-informed"),
            f"[SIMILAR]{VALSARTAN_SMARTS_PATTERN} 0.99[/SIMILAR]"
            "[CLOGP]2.02[/CLOGP][TPSA]77.04[/TPSA]",
        )

    def test_each_call_returns_fresh_mutable_specs(self):
        first = get_pmo_additional_properties("qed", "task-informed")
        second = get_pmo_additional_properties("qed", "task-informed")
        first["qed"]["value"] = "0.50"
        self.assertNotIn("value", second["qed"])

    def test_unknown_task_or_mode_fails_loudly(self):
        with self.assertRaises(ValueError):
            get_pmo_additional_properties("not_a_pmo_task", "task-informed")
        with self.assertRaises(ValueError):
            get_pmo_additional_properties("qed", "task-specific")


if __name__ == "__main__":
    unittest.main()
