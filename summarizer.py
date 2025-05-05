from flask import Flask, request, jsonify
from transformers import pipeline
from newspaper import Article

app = Flask(__name__)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    url = data.get('url', '')

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text

        if not text or len(text.split()) < 50:
            return jsonify({'error': 'Article too short or failed to parse'}), 400

        summary = summarizer(text, max_length=100, min_length=25, do_sample=False)
        return jsonify({
            'title': article.title,
            'summary': summary[0]['summary_text']
        })
    except Exception as e:
        return jsonify({'error': f'Failed to process article: {str(e)}'}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
