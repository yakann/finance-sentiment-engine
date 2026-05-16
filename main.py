import json
from pathlib import Path
from analyzer.openai_analyzer import analyze_news


def main():
    news_path = Path("cache/news.jsonl")
    items = [json.loads(line) for line in news_path.read_text().splitlines() if line.strip()][:5]

    results = []
    for item in items:
        print(f"\nAnalyzing: {item['title'][:60]}...")
        result = analyze_news(
            title=item["title"],
            summary=item["summary"],
            ticker=item["ticker"],
        )
        results.append(result.model_dump())
        print(json.dumps(result.model_dump(), indent=2))

    output_path = Path("cache/analysis.jsonl")
    with output_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
