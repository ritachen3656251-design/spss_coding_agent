import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from itertools import combinations


class CaseRepository:
    """
    疑难案例库：
    1. 自动识别并存储疑难案例
    2. 分析案例特征，总结模式
    3. 生成判断规则建议
    """

    def __init__(self, storage_path="data/knowledge/cases.json"):
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)

        self.cases = self._load_cases()

        self.session_stats = {
            "difficult_count": 0,
            "model_conflicts": 0,
            "low_confidence_count": 0,
            "new_patterns": [],
        }

    def _load_cases(self) -> dict:
        """加载历史案例"""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)

        return {
            "difficult_cases": [],
            "ambiguous_terms": {},
            "boundary_patterns": [],
            "respondent_tendencies": {},
            "metadata": {
                "total_cases": 0,
                "last_updated": None,
            },
        }

    def save_cases(self):
        """持久化案例库"""
        self.cases["metadata"]["last_updated"] = datetime.now().isoformat()
        self.cases["metadata"]["total_cases"] = len(self.cases["difficult_cases"])

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.cases, f, ensure_ascii=False, indent=2)

    def add_difficult_case(self, case_data: dict):
        """
        添加疑难案例

        case_data格式：
        {
            "id": "编号",
            "text": "回答内容",
            "score": 打分结果,
            "confidence": 置信度,
            "conflict_details": {...},  # 如果双模型不一致
            "timestamp": "时间戳",
            "tags": ["标签1", "标签2"]  # 自动生成的特征标签
        }
        """
        if not self._is_difficult(case_data):
            return False

        tags = self._auto_tag_case(case_data["text"], case_data)
        case_data["tags"] = tags

        if not self._is_duplicate(case_data):
            self.cases["difficult_cases"].append(case_data)
            self.session_stats["difficult_count"] += 1

            if case_data.get("score") in (0, 1, 2):
                self._update_ambiguous_terms(case_data["text"], case_data["score"])

            return True

        return False

    def _is_difficult(self, case_data: dict) -> bool:
        """
        判断是否为疑难案例
        """
        if case_data.get("confidence", 1) < 0.6:
            return True

        if "conflict_details" in case_data:
            return True

        if case_data.get("iteration_count", 1) > 1:
            return True

        if case_data.get("score") == 1:
            if case_data.get("confidence", 0) < 0.75:
                return True

        return False

    def _auto_tag_case(self, text: str, case_data: dict) -> list:
        """自动为案例打标签"""
        tags = []

        if len(text) < 5:
            tags.append("极短文本")
        elif len(text) < 10:
            tags.append("短文本")
        elif len(text) > 30:
            tags.append("长文本")

        ambiguous_words = ["想", "可能", "好像", "应该", "似乎", "大概"]
        found = [w for w in ambiguous_words if w in text]
        if found:
            tags.append("模糊表述")
            tags.append(f"含模糊词:{found}")

        has_belief_word = any(w in text for w in ["以为", "认为", "觉得"])
        has_action_word = any(w in text for w in ["拿", "捡", "抓", "看"])

        if has_belief_word and has_action_word:
            tags.append("信念+行动混合")
        elif has_belief_word:
            tags.append("纯信念表述")
        elif has_action_word:
            tags.append("纯行动表述")

        if "不" in text or "没" in text:
            tags.append("含否定")

        score = case_data.get("score")
        confidence = case_data.get("confidence", 0)

        if score == 1:
            tags.append("边界分数(1分)")
        if confidence < 0.5:
            tags.append("极低置信度")
        elif confidence < 0.7:
            tags.append("低置信度")

        if "conflict_details" in case_data:
            conflict = case_data["conflict_details"]
            primary_score = conflict.get("primary_score")
            secondary_score = conflict.get("secondary_score")

            if primary_score is not None and secondary_score is not None:
                tags.append(f"模型冲突({primary_score}vs{secondary_score})")

        return tags

    def _is_duplicate(self, case_data: dict) -> bool:
        """检查是否重复（基于文本相似度）"""
        text = case_data.get("text", "")
        for existing_case in self.cases["difficult_cases"]:
            similarity = SequenceMatcher(
                None,
                text,
                existing_case.get("text", ""),
            ).ratio()

            if similarity > 0.9:
                return True

        return False

    def _update_ambiguous_terms(self, text: str, score: int):
        """统计模糊词汇与分数的关系"""
        words = re.findall(r"[\u4e00-\u9fff]+", text)

        for word in words:
            if len(word) < 2:
                continue

            if word not in self.cases["ambiguous_terms"]:
                self.cases["ambiguous_terms"][word] = {
                    "count": 0,
                    "score_distribution": {0: 0, 1: 0, 2: 0},
                    "is_ambiguous": False,
                }

            if score in (0, 1, 2):
                self.cases["ambiguous_terms"][word]["count"] += 1
                self.cases["ambiguous_terms"][word]["score_distribution"][
                    score
                ] += 1

            dist = self.cases["ambiguous_terms"][word]["score_distribution"]
            non_zero_scores = sum(1 for v in dist.values() if v > 0)

            if (
                non_zero_scores >= 2
                and self.cases["ambiguous_terms"][word]["count"] >= 3
            ):
                self.cases["ambiguous_terms"][word]["is_ambiguous"] = True

    def get_ambiguous_terms_report(self) -> str:
        """生成模糊词汇报告"""
        ambiguous = {
            k: v
            for k, v in self.cases["ambiguous_terms"].items()
            if v.get("is_ambiguous") and v["count"] >= 3
        }

        if not ambiguous:
            return "暂未发现明显的模糊词汇。"

        report = "## 模糊词汇分析\n\n"
        report += "以下词汇在不同分数段都出现过，容易导致混淆：\n\n"

        sorted_terms = sorted(
            ambiguous.items(), key=lambda x: x[1]["count"], reverse=True
        )

        for word, stats in sorted_terms[:10]:
            dist = stats["score_distribution"]
            total = stats["count"]
            if total == 0:
                continue

            report += f"### {word} (出现{total}次)\n"
            report += f"- 0分: {dist.get(0, 0)}次 ({dist.get(0, 0)/total*100:.0f}%)\n"
            report += f"- 1分: {dist.get(1, 0)}次 ({dist.get(1, 0)/total*100:.0f}%)\n"
            report += f"- 2分: {dist.get(2, 0)}次 ({dist.get(2, 0)/total*100:.0f}%)\n"

            max_score = max(dist, key=dist.get)
            if dist[max_score] / total > 0.6:
                report += f"**建议**: '{word}'更倾向于{max_score}分，但仍需结合上下文\n\n"
            else:
                report += f"**建议**: '{word}'歧义较大，需特别关注上下文\n\n"

        return report

    def analyze_boundary_patterns(self) -> dict:
        """分析边界模式：1分和2分的区别在哪里？"""
        score_1_cases = [
            c for c in self.cases["difficult_cases"] if c.get("score") == 1
        ]
        score_2_cases = [
            c for c in self.cases["difficult_cases"] if c.get("score") == 2
        ]

        if len(score_1_cases) < 5 or len(score_2_cases) < 5:
            return {"status": "insufficient_data"}

        def extract_keywords(cases):
            all_words = []
            for case in cases:
                words = re.findall(r"[\u4e00-\u9fff]+", case.get("text", ""))
                all_words.extend([w for w in words if len(w) >= 2])
            return Counter(all_words)

        words_1 = extract_keywords(score_1_cases)
        words_2 = extract_keywords(score_2_cases)

        unique_to_1 = {}
        for word, count in words_1.most_common(20):
            if word not in words_2 or words_2[word] < count / 2:
                unique_to_1[word] = count

        unique_to_2 = {}
        for word, count in words_2.most_common(20):
            if word not in words_1 or words_1[word] < count / 2:
                unique_to_2[word] = count

        return {
            "status": "success",
            "score_1_indicators": dict(list(unique_to_1.items())[:10]),
            "score_2_indicators": dict(list(unique_to_2.items())[:10]),
            "total_1_cases": len(score_1_cases),
            "total_2_cases": len(score_2_cases),
        }

    def get_boundary_patterns_report(self) -> str:
        """生成边界模式报告"""
        analysis = self.analyze_boundary_patterns()

        if analysis["status"] == "insufficient_data":
            return "疑难案例数量不足，暂无法分析边界模式（需要至少各5个1分和2分案例）。"

        report = "## 1分与2分的边界特征\n\n"
        report += f"基于{analysis['total_1_cases']}个1分案例和{analysis['total_2_cases']}个2分案例的分析：\n\n"

        report += "### 1分案例的典型词汇\n"
        for word, count in analysis["score_1_indicators"].items():
            report += f"- **{word}** (出现{count}次)\n"

        report += "\n### 2分案例的典型词汇\n"
        for word, count in analysis["score_2_indicators"].items():
            report += f"- **{word}** (出现{count}次)\n"

        report += "\n### 判断建议\n"
        report += "- 如果回答主要包含1分词汇（如'想'、'打算'），倾向于1分\n"
        report += "- 如果回答主要包含2分词汇（如'以为'、'没发现'），倾向于2分\n"
        report += "- 边界case建议结合情景描述判断\n\n"

        return report

    def analyze_respondent_tendencies(self, all_responses: list) -> dict:
        """分析回答者倾向"""
        valid = [r for r in all_responses if r.get("score") != 999]
        if len(valid) < 50:
            return {"status": "insufficient_data"}

        score_dist = Counter([r["score"] for r in valid])

        lengths = [len(r.get("text", "")) for r in valid]
        avg_length = sum(lengths) / len(lengths) if lengths else 0

        mental_words = ["想", "以为", "认为", "觉得", "打算", "希望"]
        mental_count = sum(
            1 for r in valid if any(w in r.get("text", "") for w in mental_words)
        )
        mental_rate = mental_count / len(valid)

        descriptive_words = ["因为", "所以", "然后", "结果", "发现"]
        detailed_count = sum(
            1
            for r in valid
            if any(w in r.get("text", "") for w in descriptive_words)
        )
        detail_rate = detailed_count / len(valid)

        return {
            "status": "success",
            "total_responses": len(valid),
            "score_distribution": dict(score_dist),
            "avg_response_length": round(avg_length, 1),
            "mentalization_rate": round(mental_rate, 3),
            "detail_rate": round(detail_rate, 3),
            "interpretation": self._interpret_tendencies(
                score_dist, avg_length, mental_rate, detail_rate
            ),
        }

    def _interpret_tendencies(
        self, score_dist, avg_length, mental_rate, detail_rate
    ) -> str:
        """解释回答者倾向"""
        total = sum(score_dist.values()) or 1

        interpretation = []

        if mental_rate > 0.6:
            interpretation.append(
                "被试群体整体心理化水平较高，擅长理解他人心理状态"
            )
        elif mental_rate < 0.3:
            interpretation.append(
                "被试群体心理化水平较低，更倾向于描述客观事实"
            )
        else:
            interpretation.append("被试群体心理化水平中等")

        if score_dist.get(2, 0) / total > 0.5:
            interpretation.append("超过半数回答达到2分标准，说明整体理解较好")
        elif score_dist.get(0, 0) / total > 0.4:
            interpretation.append("较多回答未能识别心理状态，可能需要额外引导")

        if avg_length < 8:
            interpretation.append("回答普遍较短，可能信息量不足")
        elif avg_length > 20:
            interpretation.append("回答普遍详细，包含较多情景描述")

        return "；".join(interpretation) + "。"

    def get_respondent_report(self, all_responses: list) -> str:
        """生成回答者倾向报告"""
        analysis = self.analyze_respondent_tendencies(all_responses)

        if analysis["status"] == "insufficient_data":
            return "样本量不足，暂无法分析回答者倾向（需要至少50个有效回答）。"

        report = "## 回答者群体倾向分析\n\n"
        report += f"基于{analysis['total_responses']}个有效回答的分析：\n\n"

        report += "### 分数分布\n"
        dist = analysis["score_distribution"]
        for score in [0, 1, 2]:
            count = dist.get(score, 0)
            pct = count / analysis["total_responses"] * 100
            report += f"- {score}分: {count}人 ({pct:.1f}%)\n"

        report += f"\n### 回答特征\n"
        report += f"- 平均回答长度: {analysis['avg_response_length']}字\n"
        report += f"- 心理化词汇使用率: {analysis['mentalization_rate']*100:.1f}%\n"
        report += f"- 详细描述使用率: {analysis['detail_rate']*100:.1f}%\n"

        report += f"\n### 整体解释\n"
        report += analysis["interpretation"] + "\n\n"

        report += "### 建议\n"
        report += "- 如果心理化水平低，可以在指导语中强调\"理解他人想法\"\n"
        report += "- 如果回答过短，可以提示\"请详细描述你的理解\"\n"

        return report

    def generate_learned_rules(self) -> list:
        """基于疑难案例，生成新的判断规则建议"""
        if len(self.cases["difficult_cases"]) < 20:
            return []

        rules = []

        ambiguous = {
            k: v
            for k, v in self.cases["ambiguous_terms"].items()
            if v.get("is_ambiguous") and v["count"] >= 5
        }

        for word, stats in list(ambiguous.items())[:5]:
            dist = stats["score_distribution"]
            total = stats["count"]
            if total == 0:
                continue

            dominant_score = max(dist, key=dist.get)
            dominant_rate = dist[dominant_score] / total

            if dominant_rate > 0.7:
                rules.append({
                    "type": "word_tendency",
                    "word": word,
                    "suggested_score": dominant_score,
                    "confidence": round(dominant_rate, 2),
                    "evidence_count": total,
                    "rule_text": f"当回答包含'{word}'时，有{dominant_rate*100:.0f}%概率为{dominant_score}分",
                })

        boundary = self.analyze_boundary_patterns()
        if boundary["status"] == "success":
            for word, count in list(boundary["score_2_indicators"].items())[:3]:
                if count >= 5:
                    rules.append({
                        "type": "boundary_indicator",
                        "word": word,
                        "suggested_score": 2,
                        "confidence": 0.8,
                        "evidence_count": count,
                        "rule_text": f"'{word}'是2分的强指示词（出现{count}次）",
                    })

        two_score_texts = [
            c["text"]
            for c in self.cases["difficult_cases"]
            if c.get("score") == 2
        ]

        if len(two_score_texts) >= 10:
            word_pairs = Counter()

            for text in two_score_texts:
                words = re.findall(r"[\u4e00-\u9fff]{2,}", text or "")
                for w1, w2 in combinations(words, 2):
                    if w1 != w2:
                        pair = tuple(sorted([w1, w2]))
                        word_pairs[pair] += 1

            for pair, count in word_pairs.most_common(3):
                if count >= 3:
                    rules.append({
                        "type": "combination_pattern",
                        "words": list(pair),
                        "suggested_score": 2,
                        "confidence": 0.75,
                        "evidence_count": count,
                        "rule_text": f"'{pair[0]}'和'{pair[1]}'同时出现时，通常为2分",
                    })

        return rules

    def get_learned_rules_report(self) -> str:
        """生成学到的规则报告"""
        rules = self.generate_learned_rules()

        if not rules:
            return "疑难案例数量不足，暂未学习到新规则（需要至少20个疑难案例）。"

        report = "## Agent学习到的新规则\n\n"
        report += f"基于{len(self.cases['difficult_cases'])}个疑难案例的分析，Agent总结出以下规则：\n\n"

        by_type = defaultdict(list)
        for rule in rules:
            by_type[rule["type"]].append(rule)

        if "word_tendency" in by_type:
            report += "### 词汇倾向规则\n"
            for rule in by_type["word_tendency"]:
                report += f"- **{rule['rule_text']}** (基于{rule['evidence_count']}个案例)\n"
            report += "\n"

        if "boundary_indicator" in by_type:
            report += "### 边界指示词\n"
            for rule in by_type["boundary_indicator"]:
                report += f"- **{rule['rule_text']}**\n"
            report += "\n"

        if "combination_pattern" in by_type:
            report += "### 组合模式\n"
            for rule in by_type["combination_pattern"]:
                report += f"- **{rule['rule_text']}** (出现{rule['evidence_count']}次)\n"
            report += "\n"

        report += "### 如何应用这些规则\n"
        report += "1. 这些规则可以加入到关键词匹配工具中作为补充\n"
        report += "2. 在评分标准中明确这些模式\n"
        report += "3. 用于训练新的评分者\n"

        return report
