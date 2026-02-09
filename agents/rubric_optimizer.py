"""
评分标准优化器：分析疑难案例、生成改进建议、A/B 测试、版本化管理
"""
from agents.base_agent import BaseAgent
from agents.case_repository import CaseRepository
import json
import os
import re
from datetime import datetime


class RubricOptimizer(BaseAgent):
    """
    评分标准优化器：
    1. 分析疑难案例找出标准模糊点
    2. 使用LLM生成标准改进建议
    3. 自动测试新标准效果
    4. 版本化管理评分标准
    """

    def __init__(self, rubric: str, case_repo: CaseRepository):
        super().__init__(name="RubricOptimizer")
        self.original_rubric = rubric
        self.case_repo = case_repo
        self.version_dir = "data/rubrics/versions"
        os.makedirs(self.version_dir, exist_ok=True)

        # 加载标准版本历史
        self.versions = self._load_version_history()

    def _load_version_history(self) -> dict:
        """加载评分标准的版本历史"""
        history_file = os.path.join(self.version_dir, "history.json")

        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # 初始化：保存原始版本
        initial_version = {
            "v1.0": {
                "content": self.original_rubric,
                "created_at": datetime.now().isoformat(),
                "changes": "初始版本",
                "performance": None,
            }
        }

        # 持久化初始版本
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(initial_version, f, ensure_ascii=False, indent=2)

        return initial_version

    def analyze_rubric_weaknesses(self) -> dict:
        """
        分析评分标准的薄弱点

        返回：
        {
            "ambiguous_areas": ["模糊点1", "模糊点2"],
            "missing_rules": ["缺失的规则1", "缺失的规则2"],
            "conflicting_cases": [案例1, 案例2],
            "suggested_improvements": "改进建议"
        }
        """
        # 1. 收集疑难案例
        difficult_cases = self.case_repo.cases.get("difficult_cases", [])

        if len(difficult_cases) < 10:
            return {
                "status": "insufficient_data",
                "message": "疑难案例不足10个，暂无法分析",
            }

        # 2. 按问题类型分组
        low_confidence_cases = [c for c in difficult_cases if c.get("confidence", 1) < 0.6]
        conflict_cases = [c for c in difficult_cases if "conflict_details" in c]
        boundary_cases = [c for c in difficult_cases if c.get("score") == 1]

        # 3. 让LLM分析问题
        analysis_prompt = f"""你是评分标准专家。请分析以下疑难案例，找出当前评分标准的问题。

当前评分标准：
{self.original_rubric}

疑难案例示例（低置信度）：
"""
        for case in low_confidence_cases[:5]:
            analysis_prompt += f"- 文本：{case.get('text', '')}\n  打分：{case.get('score')}分\n  置信度：{case.get('confidence', 0)}\n\n"

        analysis_prompt += "\n模型冲突案例：\n"
        for case in conflict_cases[:3]:
            conflict = case.get("conflict_details", {})
            analysis_prompt += f"- 文本：{case.get('text', '')}\n  模型A：{conflict.get('primary_score')}分\n  模型B：{conflict.get('secondary_score')}分\n\n"

        analysis_prompt += """
请输出JSON格式的分析：
{
    "ambiguous_areas": ["标准中哪些部分表述模糊"],
    "missing_rules": ["应该补充什么规则"],
    "common_confusion": "最常见的混淆点是什么",
    "suggested_improvements": "具体改进建议"
}
"""

        system_prompt = "你是评分标准专家，擅长发现标准的问题并提出改进建议。只返回JSON。"

        result = self.call_llm_with_json(system_prompt, analysis_prompt)

        if result["success"]:
            analysis = result["data"]
            analysis["stats"] = {
                "total_difficult": len(difficult_cases),
                "low_confidence": len(low_confidence_cases),
                "conflicts": len(conflict_cases),
                "boundary": len(boundary_cases),
            }
            return analysis
        else:
            return {"status": "analysis_failed"}

    def generate_improved_rubric(self, analysis: dict) -> dict:
        """
        基于分析生成改进的评分标准

        返回：
        {
            "new_rubric": "改进后的标准",
            "changes": ["修改点1", "修改点2"],
            "reasoning": "为什么这样改"
        }
        """
        if analysis.get("status") == "insufficient_data":
            return analysis

        if analysis.get("status") == "analysis_failed":
            return {"status": "generation_failed"}

        improvement_prompt = f"""你是评分标准优化专家。请根据以下分析改进评分标准。

原始评分标准：
{self.original_rubric}

发现的问题：
- 模糊区域：{analysis.get('ambiguous_areas', [])}
- 缺失规则：{analysis.get('missing_rules', [])}
- 常见混淆：{analysis.get('common_confusion', '未知')}

改进建议：
{analysis.get('suggested_improvements', '')}

学到的规则（来自案例库）：
"""
        # 加入Agent学到的规则
        learned_rules = self.case_repo.generate_learned_rules()
        for rule in learned_rules[:5]:
            improvement_prompt += f"- {rule.get('rule_text', '')}\n"

        improvement_prompt += """
请生成改进后的评分标准，要求：
1. 保持原有结构
2. 针对模糊点增加明确说明
3. 补充缺失的规则
4. 加入实际案例作为参考

输出JSON格式：
{
    "new_rubric": "完整的改进后标准（markdown格式）",
    "changes": ["修改点1：具体改了什么", "修改点2：..."],
    "reasoning": "为什么这样改，预期解决什么问题"
}
"""

        system_prompt = "你是评分标准专家。生成清晰、可操作的评分标准。"

        result = self.call_llm_with_json(system_prompt, improvement_prompt)

        if result["success"]:
            return result["data"]
        else:
            return {"status": "generation_failed"}

    def save_new_version(self, improved_rubric: dict) -> str:
        """
        保存新版本的评分标准

        返回：版本号（如 "v1.1"）
        """
        # 确定新版本号
        current_versions = list(self.versions.keys())
        latest_version = max(current_versions, key=lambda v: tuple(map(int, v[1:].split("."))))
        major, minor = map(int, latest_version[1:].split("."))
        new_version = f"v{major}.{minor + 1}"

        # 保存新版本
        changes = improved_rubric.get("changes", [])
        if isinstance(changes, str):
            changes = [changes]

        self.versions[new_version] = {
            "content": improved_rubric["new_rubric"],
            "created_at": datetime.now().isoformat(),
            "changes": changes,
            "reasoning": improved_rubric.get("reasoning", ""),
            "performance": None,  # 将在A/B测试后填充
        }

        # 持久化
        history_file = os.path.join(self.version_dir, "history.json")
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(self.versions, f, ensure_ascii=False, indent=2)

        # 保存单独文件
        version_file = os.path.join(self.version_dir, f"rubric_{new_version}.md")
        with open(version_file, "w", encoding="utf-8") as f:
            f.write(improved_rubric["new_rubric"])

        self.log(f"✓ 新版本已保存：{new_version}")
        return new_version

    def ab_test_rubrics(self, test_cases: list, old_version: str, new_version: str) -> dict:
        """
        A/B测试：对比新旧标准的效果

        test_cases: [{"text": "...", "ground_truth": 2}, ...]

        返回：
        {
            "old_performance": {"accuracy": 0.85, "avg_confidence": 0.75},
            "new_performance": {"accuracy": 0.90, "avg_confidence": 0.82},
            "improvement": "+5%",
            "recommendation": "建议采用新版本"
        }
        """
        if len(test_cases) < 20:
            return {"status": "insufficient_test_cases"}

        if old_version not in self.versions or new_version not in self.versions:
            return {"status": "version_not_found"}

        from tools.llm_client import LLMClient

        llm = LLMClient(model_type="qwen")

        def test_rubric(rubric_content, cases):
            correct = 0
            confidences = []

            for case in cases:
                prompt = f"""评分标准：
{rubric_content}

回答：{case.get('text', '')}

请打分（0/1/2）并给出置信度。
输出JSON：{{"score": X, "confidence": 0.X}}"""

                result = llm.call("你是评分专家。只返回JSON。", prompt)
                if result.get("success") and result.get("response"):
                    try:
                        raw = result["response"].strip()
                        json_match = re.search(r"\{[\s\S]*\}", raw)
                        if json_match:
                            data = json.loads(json_match.group())
                            if data.get("score") == case.get("ground_truth"):
                                correct += 1
                            if "confidence" in data:
                                confidences.append(data["confidence"])
                    except (json.JSONDecodeError, TypeError):
                        pass

            return {
                "accuracy": correct / len(cases) if cases else 0,
                "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            }

        # 测试旧版本
        old_rubric = self.versions[old_version]["content"]
        old_perf = test_rubric(old_rubric, test_cases)

        # 测试新版本
        new_rubric = self.versions[new_version]["content"]
        new_perf = test_rubric(new_rubric, test_cases)

        # 对比
        improvement = (
            (new_perf["accuracy"] - old_perf["accuracy"]) / old_perf["accuracy"] * 100
            if old_perf["accuracy"] > 0
            else 0
        )

        recommendation = "建议采用新版本" if improvement > 2 else "保持旧版本"

        return {
            "old_performance": old_perf,
            "new_performance": new_perf,
            "improvement_pct": round(improvement, 2),
            "recommendation": recommendation,
        }

    def optimize_workflow(self):
        """
        完整的优化工作流
        """
        self.log("开始评分标准优化流程...")

        # 1. 分析问题
        self.log("步骤1：分析当前标准的薄弱点")
        analysis = self.analyze_rubric_weaknesses()

        if analysis.get("status") == "insufficient_data":
            self.log("⚠️  疑难案例不足，无法进行优化")
            return None

        if analysis.get("status") == "analysis_failed":
            self.log("❌ 分析失败")
            return None

        self.log(f"发现{len(analysis.get('ambiguous_areas', []))}个模糊点")

        # 2. 生成改进版本
        self.log("步骤2：生成改进的评分标准")
        improved = self.generate_improved_rubric(analysis)

        if improved.get("status") == "generation_failed":
            self.log("❌ 生成失败")
            return None

        self.log(f"生成了{len(improved.get('changes', []))}处改进")

        # 3. 保存新版本
        self.log("步骤3：保存新版本")
        new_version = self.save_new_version(improved)

        # 4. 生成报告
        report = self._generate_optimization_report(analysis, improved, new_version)

        return {
            "new_version": new_version,
            "analysis": analysis,
            "improvements": improved,
            "report": report,
        }

    def _generate_optimization_report(
        self, analysis: dict, improved: dict, version: str
    ) -> str:
        """
        生成优化报告
        """
        stats = analysis.get("stats", {})
        report = f"""# 评分标准优化报告

版本：{version}
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 一、当前标准存在的问题

### 统计数据
- 疑难案例总数：{stats.get('total_difficult', 0)}
- 低置信度案例：{stats.get('low_confidence', 0)}
- 模型冲突案例：{stats.get('conflicts', 0)}
- 边界案例（1分）：{stats.get('boundary', 0)}

### 发现的问题

#### 模糊区域
"""
        for area in analysis.get("ambiguous_areas", []):
            report += f"- {area}\n"

        report += "\n#### 缺失规则\n"
        for rule in analysis.get("missing_rules", []):
            report += f"- {rule}\n"

        report += f"\n#### 最常见混淆\n{analysis.get('common_confusion', '未识别')}\n"

        report += "\n---\n\n## 二、改进内容\n\n"
        for i, change in enumerate(improved.get("changes", []), 1):
            report += f"{i}. {change}\n"

        report += f"\n### 改进理由\n{improved.get('reasoning', '')}\n"

        report += "\n---\n\n## 三、新版本标准\n\n"
        report += improved.get("new_rubric", "")

        report += "\n\n---\n\n## 四、如何使用新版本\n\n"
        report += f"新版本已保存至：`data/rubrics/versions/rubric_{version}.md`\n\n"
        report += "使用方法：\n"
        report += "1. 在 `prompts/scoring_rubric.txt` 中替换为新版本内容\n"
        report += "2. 或在config.py中指定版本号\n"
        report += "3. 重新运行系统进行测试\n\n"
        report += "⚠️  建议先用A/B测试验证新版本效果后再全面采用\n"

        return report
