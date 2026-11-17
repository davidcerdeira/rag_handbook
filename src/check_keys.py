from dotenv import load_dotenv
import anthropic, voyageai

load_dotenv()  # reads .env into environment variables


# Claude: tiny request with the cheap model
claude = anthropic.Anthropic()
msg = claude.messages.create(
    model="claude-haiku-4-5",
    max_tokens=20,
    messages=[{"role": "user", "content": "Say 'Claude OK'"}],
)
print(msg.content[0].text)

vo = voyageai.Client()
result = vo.embed(["hello world"], model="voyage-4", input_type="document")
print("Voyage OK, vector length:", len(result.embeddings[0]))