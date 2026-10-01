from retrieval.retriever import retrieve
from generation.generator import generate_answer


def main():
    print("\n========================================")
    print("      ACADEMIC QUESTION ANSWERING")
    print("========================================")

    while True:
        question = input(
            "\nAsk your question (type 'exit' to stop): "
        ).strip()

        if question.lower() == "exit":
            print("\nThank you!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            print("\nSearching study material...")

            results = retrieve(question, top_k=5)

            if not results:
                print("No relevant information found.")
                continue

            print(f"Found {len(results)} relevant sections.")

            print("\nGenerating answer...")

            answer = generate_answer(question, results)

            print("\n----------------------------------------")
            print("ANSWER")
            print("----------------------------------------")
            print(answer)

            print("\n----------------------------------------")
            print("SOURCES")
            print("----------------------------------------")

            for i, result in enumerate(results, 1):
                metadata = result.get("metadata", {})

                source = metadata.get("source", "Unknown")
                page = metadata.get("page", "Unknown")
                score = result.get("score", 0)

                print(
                    f"{i}. {source} | "
                    f"Page: {page} | "
                    f"Score: {score:.4f}"
                )

        except Exception as e:
            print(f"\nSomething went wrong: {e}")


if __name__ == "__main__":
    main()