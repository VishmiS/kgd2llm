

def convert_covid_knowledge_to_sentences(covid_knowledge):
    """Lazy import to avoid circular dependencies"""
    from entity_linking.covid_handler import convert_covid_knowledge_to_sentences as convert_func
    return convert_func(covid_knowledge)


# -----------------------------
# Clean & Structured Print
# -----------------------------
def print_clean_pipeline_result(result, max_facts_per_entity=5):
    entity_labels = {**result['falcon_qids'], **result['dbpedia_entities']}

    print("\n" + "=" * 50)
    print(f"Original Query: {result['original_query']}")
    print("=" * 50)

    # --- Entities ---
    print("\n[Entities]")
    for eid, label in entity_labels.items():
        source = "Wikidata" if eid in result['falcon_qids'] else "DBpedia"
        print(f"  {source}: {label} ({eid})")

    # ========== ADD COVID KNOWLEDGE DISPLAY ==========
    if result.get("is_covid_related"):
        print("\n[COVID-19 Knowledge]")
        covid_knowledge = result.get("covid_knowledge", {})

        if not covid_knowledge:
            print("  No specialized COVID-19 knowledge found")
        else:
            # Show mutation highlights
            if "mutations" in covid_knowledge and covid_knowledge["mutations"]:
                print(f"  Mutations Found: {len(covid_knowledge['mutations'])}")
                for mutation in covid_knowledge["mutations"][:3]:  # Show top 3
                    mutation_type = mutation.get('mutation_type', 'mutation')
                    print(f"    - {mutation['name']} ({mutation_type})")

            if "spike_mutations" in covid_knowledge and covid_knowledge["spike_mutations"]:
                print(f"  Spike Mutations: {len(covid_knowledge['spike_mutations'])}")
                for mutation in covid_knowledge["spike_mutations"][:3]:
                    print(f"    - {mutation['name']}")

    # --- Merge and deduplicate facts ---
    def normalize_facts(facts_dict):
        merged = {}
        for eid, facts in facts_dict.items():
            unique = set((prop.lower(), val.lower()) for prop, val in facts)
            merged[eid] = [(prop.capitalize(), val.capitalize()) for prop, val in unique]
        return merged

    merged_wikidata = normalize_facts(result['wikidata_facts_filtered'])
    merged_dbpedia = normalize_facts(result['dbpedia_facts_filtered'])

    # --- Filtered Facts ---
    if any(merged_wikidata.values()) or any(merged_dbpedia.values()):
        print("\n[Filtered Facts]")
        for eid, facts in merged_wikidata.items():
            if facts:
                print(f"  {result['falcon_qids'][eid]} (Wikidata):")
                for prop, val in facts[:max_facts_per_entity]:
                    print(f"    - {prop}: {val}")
        for eid, facts in merged_dbpedia.items():
            if facts:
                print(f"  {result['dbpedia_entities'][eid]} (DBpedia):")
                for prop, val in facts[:max_facts_per_entity]:
                    print(f"    - {prop}: {val}")

    # --- Natural Language Summary ---
    if any(result['natural_language_summary'].values()):
        print("\n[Natural Language Summary]")
        for eid, sentences in result['natural_language_summary'].items():
            if sentences:
                label = entity_labels.get(eid, eid)
                unique_sentences = sorted(set(sentences))
                for sentence in unique_sentences[:max_facts_per_entity]:
                    print(f"  - {sentence}")

    # --- Reformulated Query (concise) ---
    top_sentences = []
    for sentences in result['natural_language_summary'].values():
        top_sentences.extend(sentences[:max_facts_per_entity])

    # ========== ADD COVID KNOWLEDGE TO REFORMULATION ==========
    if result.get("is_covid_related") and result.get("covid_knowledge"):
        covid_sentences = convert_covid_knowledge_to_sentences(result["covid_knowledge"])
        top_sentences.extend(covid_sentences[:3])  # Add top 3 COVID sentences

    reformulated_query = f"{result['original_query']} {' '.join(top_sentences)}"

    print("\n[Reformulated Query]")
    print(f"  {reformulated_query}")
    print("=" * 50)


def display_full_pipeline_result(result, max_facts_per_entity=10, show_scores=True):

    # === Core Display Utilities ===
    def print_entities():
        print("\n[Entities & Linking]")
        all_entities = {**result.get('falcon_qids', {}), **result.get('dbpedia_entities', {})}
        entity_types = result.get('entity_types', {})
        for eid, label in all_entities.items():
            src = "Wikidata" if eid in result.get('falcon_qids', {}) else "DBpedia"
            types = ", ".join(entity_types.get(eid, []))
            url = f"https://www.wikidata.org/wiki/{eid}" if src == "Wikidata" else f"http://dbpedia.org/resource/{eid}"
            print(f"  - {label} ({src}, {eid})")
            if types:
                print(f"      Types: {types}")
            print(f"      URL: {url}")

    def print_facts_section(facts_dict, entities_dict, title, source="wikidata"):
        print(f"\n[{title}]")
        for eid, facts in facts_dict.items():
            if not facts:
                continue
            elabel = entities_dict.get(eid, eid)
            print(f"→ {elabel} ({eid})")
            for f in facts[:max_facts_per_entity]:
                if isinstance(f, dict):
                    prop = f.get("property") or f.get("p")
                    val = f.get("value") or f.get("v")
                    pid = f.get("pid")
                    score = f.get("score")
                    prov = f.get("provenance", "")
                elif isinstance(f, (list, tuple)) and len(f) >= 2:
                    prop, val = f[:2]
                    pid = f[2] if len(f) > 2 else None
                    score = f[3] if len(f) > 3 else None
                    prov = f[4] if len(f) > 4 else ""
                else:
                    continue

                pid_str = f" ({pid})" if pid else ""
                prov_str = f" ← {prov}" if prov else ""
                score_str = f"  [score={score:.3f}]" if show_scores and isinstance(score, (int, float)) else ""
                prop_url = f"https://www.wikidata.org/wiki/Property:{pid}" if pid and source == "wikidata" else ""
                val_url = f"https://www.wikidata.org/wiki/{val}" if isinstance(val, str) and val.startswith("Q") else ""
                val_display = f"{val} ({val_url})" if val_url else val

                if prop_url:
                    print(f"    • {prop}{pid_str}: {val_display}{score_str}{prov_str}")
                    print(f"       → {prop_url}")
                else:
                    print(f"    • {prop}{pid_str}: {val_display}{score_str}{prov_str}")

    def print_relation_diagnostics():
        print("\n[Relations Identified]")
        rels = result.get("falcon_relations", {})
        if not rels:
            print("  (none)")
            return
        for rid, label in rels.items():
            wikidata_url = f"https://www.wikidata.org/wiki/Property:{rid}" if rid.startswith("P") else f"http://dbpedia.org/ontology/{rid}"
            print(f"  - {label} ({rid})")
            print(f"      URL: {wikidata_url}")

    # 🔍 NEW: Show which entities have facts for each relation
    def print_relation_entity_mapping():
        print("\n[Relations ↔ Entities Mapping]")
        rels = result.get("falcon_relations", {})
        if not rels:
            print("  (none)")
            return
        found_any = False
        for rid, rlabel in rels.items():
            matching_entities = []
            for eid, facts in result.get('wikidata_facts_combined', {}).items():
                if any(
                        (
                                isinstance(f, dict)
                                and any(rlabel.lower() in str(v).lower() or rid in str(v) for v in f.values())
                        ) or (
                                isinstance(f, (list, tuple))
                                and any(rlabel.lower() in str(x).lower() or rid in str(x) for x in f)
                        ) or (
                                isinstance(f, str)
                                and (rlabel.lower() in f.lower() or rid in f)
                        )
                        for f in facts
                ):
                    matching_entities.append(result['falcon_qids'].get(eid, eid))
            if matching_entities:
                found_any = True
                print(f"  - {rlabel} ({rid}) → {', '.join(matching_entities)}")
            else:
                print(f"  - {rlabel} ({rid}) → no matching facts found")
        if not found_any:
            print("  (no relation-linked entities)")

    # 🔍 NEW: Print facts that correspond directly to relations (like 'products')
    def print_relation_facts():
        print("\n[Relation-driven Facts]")
        rels = result.get("falcon_relations", {})
        if not rels:
            print("  (none)")
            return
        for rid, rlabel in rels.items():
            print(f"\n→ Relation: {rlabel} ({rid})")
            found = False
            for eid, facts in result.get('wikidata_facts_combined', {}).items():
                elabel = result['falcon_qids'].get(eid, eid)
                for f in facts:
                    prop = f[0] if isinstance(f, (list, tuple)) else f.get("property")
                    val = f[1] if isinstance(f, (list, tuple)) else f.get("value")
                    if rlabel.lower() in str(prop).lower() or rid in str(prop):
                        found = True
                        print(f"   • {elabel} → {rlabel}: {val}")
            if not found:
                print("   (no matching facts for this relation)")

    def trace_filtered_facts():
        """Trace where filtered facts came from (for debugging)"""
        print("\n[Trace: Filtered Fact Origins]")

        # Check Wikidata facts
        for qid, label in result.get("falcon_qids", {}).items():
            filtered_facts = result.get("wikidata_facts_filtered", {}).get(qid, [])
            combined_facts = result.get("wikidata_facts_combined", {}).get(qid, [])

            for fact in filtered_facts:
                if isinstance(fact, dict):
                    prop = fact.get("property", "")
                    val = fact.get("value", "")

                    # Try to find where this fact came from
                    found_in = []

                    # Check combined facts (handle both dict and tuple formats)
                    for f in combined_facts:
                        if isinstance(f, dict) and f.get("property") == prop and f.get("value") == val:
                            found_in.append("combined")
                            break
                        elif isinstance(f, tuple) and len(f) >= 2 and f[0] == prop and f[1] == val:
                            found_in.append("combined")
                            break

                    # Check 2-hop facts
                    for f in result.get("wikidata_facts_2hop", {}).get(qid, []):
                        if isinstance(f, dict) and f.get("property") == prop and f.get("value") == val:
                            found_in.append("2-hop")
                            break
                        elif isinstance(f, tuple) and len(f) >= 2 and f[0] == prop and f[1] == val:
                            found_in.append("2-hop")
                            break

                    # Check 1-hop facts
                    for f in result.get("wikidata_facts", {}).get(qid, []):
                        if isinstance(f, dict) and f.get("property") == prop and f.get("value") == val:
                            found_in.append("1-hop")
                            break
                        elif isinstance(f, tuple) and len(f) >= 2 and f[0] == prop and f[1] == val:
                            found_in.append("1-hop")
                            break

                    source = " + ".join(found_in) if found_in else "unknown"
                    print(f"  {label} • [{prop}] {val}  ← came from {source}")

                elif isinstance(fact, tuple) and len(fact) >= 2:
                    prop, val = fact[0], fact[1]
                    # Similar logic for tuple format
                    found_in = []

                    # Check combined facts
                    for f in combined_facts:
                        if isinstance(f, tuple) and len(f) >= 2 and f[0] == prop and f[1] == val:
                            found_in.append("combined")
                            break
                        elif isinstance(f, dict) and f.get("property") == prop and f.get("value") == val:
                            found_in.append("combined")
                            break

                    # Check 2-hop facts
                    for f in result.get("wikidata_facts_2hop", {}).get(qid, []):
                        if isinstance(f, tuple) and len(f) >= 2 and f[0] == prop and f[1] == val:
                            found_in.append("2-hop")
                            break
                        elif isinstance(f, dict) and f.get("property") == prop and f.get("value") == val:
                            found_in.append("2-hop")
                            break

                    # Check 1-hop facts
                    for f in result.get("wikidata_facts", {}).get(qid, []):
                        if isinstance(f, tuple) and len(f) >= 2 and f[0] == prop and f[1] == val:
                            found_in.append("1-hop")
                            break
                        elif isinstance(f, dict) and f.get("property") == prop and f.get("value") == val:
                            found_in.append("1-hop")
                            break

                    source = " + ".join(found_in) if found_in else "unknown"
                    print(f"  {label} • [{prop}] {val}  ← came from {source}")

        # Check DBpedia facts
        for uri, label in result.get("dbpedia_entities", {}).items():
            filtered_facts = result.get("dbpedia_facts_filtered", {}).get(uri, [])
            combined_facts = result.get("dbpedia_facts_combined", {}).get(uri, [])

            for fact in filtered_facts:
                if isinstance(fact, dict):
                    prop = fact.get("property", "")
                    val = fact.get("value", "")

                    found_in = []

                    # Check combined facts
                    for f in combined_facts:
                        if isinstance(f, dict) and f.get("property") == prop and f.get("value") == val:
                            found_in.append("combined")
                            break
                        elif isinstance(f, tuple) and len(f) >= 2 and f[0] == prop and f[1] == val:
                            found_in.append("combined")
                            break

                    # Check 2-hop facts
                    for f in result.get("dbpedia_facts_2hop", {}).get(uri, []):
                        if isinstance(f, dict) and f.get("property") == prop and f.get("value") == val:
                            found_in.append("2-hop")
                            break
                        elif isinstance(f, tuple) and len(f) >= 2 and f[0] == prop and f[1] == val:
                            found_in.append("2-hop")
                            break

                    # Check 1-hop facts
                    for f in result.get("dbpedia_facts_raw", {}).get(uri, []):
                        if isinstance(f, dict) and f.get("property") == prop and f.get("value") == val:
                            found_in.append("1-hop")
                            break
                        elif isinstance(f, tuple) and len(f) >= 2 and f[0] == prop and f[1] == val:
                            found_in.append("1-hop")
                            break

                    source = " + ".join(found_in) if found_in else "unknown"
                    print(f"  {label} • [{prop}] {val}  ← came from {source}")

                elif isinstance(fact, tuple) and len(fact) >= 2:
                    prop, val = fact[0], fact[1]
                    found_in = []

                    # Check combined facts
                    for f in combined_facts:
                        if isinstance(f, tuple) and len(f) >= 2 and f[0] == prop and f[1] == val:
                            found_in.append("combined")
                            break
                        elif isinstance(f, dict) and f.get("property") == prop and f.get("value") == val:
                            found_in.append("combined")
                            break

                    # Check 2-hop facts
                    for f in result.get("dbpedia_facts_2hop", {}).get(uri, []):
                        if isinstance(f, tuple) and len(f) >= 2 and f[0] == prop and f[1] == val:
                            found_in.append("2-hop")
                            break
                        elif isinstance(f, dict) and f.get("property") == prop and f.get("value") == val:
                            found_in.append("2-hop")
                            break

                    # Check 1-hop facts
                    for f in result.get("dbpedia_facts_raw", {}).get(uri, []):
                        if isinstance(f, tuple) and len(f) >= 2 and f[0] == prop and f[1] == val:
                            found_in.append("1-hop")
                            break
                        elif isinstance(f, dict) and f.get("property") == prop and f.get("value") == val:
                            found_in.append("1-hop")
                            break

                    source = " + ".join(found_in) if found_in else "unknown"
                    print(f"  {label} • [{prop}] {val}  ← came from {source}")

    def print_summary():
        print("\n[Natural Language Summary]")
        summary = result.get('natural_language_summary', {})
        for eid, sentences in summary.items():
            label = result['falcon_qids'].get(eid, result['dbpedia_entities'].get(eid, eid))
            print(f"→ {label}")
            for s in sentences:
                relation_mark = ""
                for rel_label in result.get("falcon_relations", {}).values():
                    if rel_label.lower() in s.lower():
                        relation_mark = "  🔍 [relation-based]"
                        break
                print(f"   • {s}{relation_mark}")

    def print_reformulated_query():
        print("\n[Reformulated Query]")
        rq = result.get('reformulated_query', '')
        if any(rel.lower() in rq.lower() for rel in result.get("falcon_relations", {}).values()):
            print(f"  {rq}")
            print("  🔍 Relation terms incorporated into reformulated query.")
        else:
            print(f"  {rq}")

    # === Display Order ===
    print("=" * 100)
    print(f"Original Query:\n  {result.get('original_query', '')}")
    print("=" * 100)

    print_entities()
    print_relation_diagnostics()
    print_relation_entity_mapping()   # 🔍 NEW
    print_relation_facts()            # 🔍 NEW

    print_facts_section(result.get('wikidata_facts', {}), result.get('falcon_qids', {}), "Wikidata Facts (1-hop)", "wikidata")
    print_facts_section(result.get('wikidata_facts_2hop', {}), result.get('falcon_qids', {}), "Wikidata Facts (2-hop)", "wikidata")
    print_facts_section(result.get('wikidata_facts_combined', {}), result.get('falcon_qids', {}), "Wikidata Facts (Combined)", "wikidata")
    print_facts_section(result.get('wikidata_facts_filtered', {}), result.get('falcon_qids', {}), "Wikidata Facts (Filtered)", "wikidata")

    print_facts_section(result.get('dbpedia_facts_raw', {}), result.get('dbpedia_entities', {}), "DBpedia Facts (1-hop)", "dbpedia")
    print_facts_section(result.get('dbpedia_facts_2hop', {}), result.get('dbpedia_entities', {}), "DBpedia Facts (2-hop)", "dbpedia")
    print_facts_section(result.get('dbpedia_facts_combined', {}), result.get('dbpedia_entities', {}), "DBpedia Facts (Combined)", "dbpedia")
    print_facts_section(result.get('dbpedia_facts_filtered', {}), result.get('dbpedia_entities', {}), "DBpedia Facts (Filtered)", "dbpedia")

    trace_filtered_facts()

    # ========== ADD COVID KNOWLEDGE DISPLAY ==========
    if result.get("is_covid_related"):
        print("\n[COVID-19 Knowledge Extracted]")
        covid_knowledge = result.get("covid_knowledge", {})

        if not covid_knowledge:
            print("  No specialized COVID-19 knowledge found")
        else:
            total_facts = sum(len(items) for items in covid_knowledge.values())
            print(f"  Found {total_facts} COVID-19 facts across {len(covid_knowledge)} categories")

            # Display mutations if available
            if "mutations" in covid_knowledge and covid_knowledge["mutations"]:
                print(f"\n  → SARS-CoV-2 Mutations Found ({len(covid_knowledge['mutations'])}):")
                for i, mutation in enumerate(covid_knowledge["mutations"][:10], 1):  # Show first 10
                    mutation_type = mutation.get('mutation_type', 'mutation')
                    print(f"     {i}. {mutation['name']} ({mutation_type})")
                    if mutation.get('description'):
                        desc = mutation['description']
                        if len(desc) > 150:
                            desc = desc[:150] + "..."
                        print(f"        {desc}")
                    if mutation.get('protein'):
                        print(f"        Affects: {mutation['protein']}")
                    if mutation.get('variant'):
                        print(f"        Found in: {mutation['variant']}")
                    print()

            # Display spike mutations if available
            if "spike_mutations" in covid_knowledge and covid_knowledge["spike_mutations"]:
                print(f"  → Spike Protein Mutations ({len(covid_knowledge['spike_mutations'])}):")
                for i, mutation in enumerate(covid_knowledge["spike_mutations"][:10], 1):
                    print(f"     {i}. {mutation['name']}")
                    if mutation.get('description'):
                        desc = mutation['description']
                        if len(desc) > 150:
                            desc = desc[:150] + "..."
                        print(f"        {desc}")
                    print()

            # Display variants if available
            if "variants" in covid_knowledge and covid_knowledge["variants"]:
                print(f"  → SARS-CoV-2 Variants ({len(covid_knowledge['variants'])}):")
                for i, variant in enumerate(covid_knowledge["variants"][:5], 1):
                    print(f"     {i}. {variant['name']}")
                    if variant.get('description'):
                        desc = variant['description']
                        if len(desc) > 100:
                            desc = desc[:100] + "..."
                        print(f"        {desc}")
                    print()

            # Display other COVID categories briefly
            other_categories = [cat for cat in covid_knowledge.keys()
                                if cat not in ['mutations', 'spike_mutations', 'variants']
                                and covid_knowledge[cat]]
            if other_categories:
                print(f"  → Other COVID-19 Information:")
                for category in other_categories:
                    count = len(covid_knowledge[category])
                    print(f"     - {category.replace('_', ' ').title()}: {count} items")

    print_summary()
    print_reformulated_query()
    print("=" * 100)