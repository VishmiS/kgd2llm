const apiUrl = "http://localhost:5000/"; // Unified backend URL

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("predict-btn").addEventListener("click", predictLabel);
    document.getElementById("reset-btn").addEventListener("click", resetForm);
    document.getElementById("submit-btn").addEventListener("click", submitQuestion);
    document.getElementById("reset-btn2").addEventListener("click", resetQAForm);

    document.getElementById('lime-explanation-sts').style.display = 'none';
    document.getElementById('lime-explanation-qa').style.display = 'none';
});

// ========== Predict Semantic Text Similarity ==========

async function predictLabel() {
    const sentence1 = document.getElementById("sentence1").value.trim();
    const sentence2 = document.getElementById("sentence2").value.trim();

    if (!sentence1 || !sentence2) {
        alert("Please enter both sentences.");
        return;
    }

    const data = { sentence1, sentence2 };

    try {
        const response = await fetch(`${apiUrl}/sts/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (!response.ok) {
            displayErrorMessage(result.error || "Unexpected error occurred.");
            throw new Error(`Server error: ${response.status}`);
        }

        // Update prediction results
        document.getElementById("predicted-label").innerText = result.predicted_label;
        document.getElementById("confidence-score").innerText = `${result.confidence_score.toFixed(2)}%`;

        // Highlight keywords in sentences
        document.getElementById("sentence1-info").innerHTML = result.highlighted_keywords.sentence1;
        document.getElementById("sentence2-info").innerHTML = result.highlighted_keywords.sentence2;

        document.getElementById("sentence2-info").innerHTML = highlightTextWithKeywords(
            result.highlighted_keywords.sentence2,
            result.lime_explanation.keywords,
            result.lime_explanation.weights
        );
                document.getElementById("prediction-result").style.display = "block";

        if (result.lime_explanation) {
            renderLimeExplanation('lime-explanation-sts', result.lime_explanation);
        }

    } catch (error) {
        console.error(error);
        alert("An error occurred: " + error.message);
    }
}

// ========== Question Answering ==========

async function submitQuestion() {
    const question = document.getElementById("question").value.trim();
    const context = document.getElementById("context").value.trim();

    if (!question || !context) {
        alert("Please enter both question and context.");
        return;
    }

    const data = { question, context };

    try {
        const qaResult = document.getElementById("qa-result");
        qaResult.style.display = "block";
        qaResult.innerHTML = `<h2>Answer</h2><p>Loading...</p>`;

        const response = await fetch(`${apiUrl}/ir/generate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (!response.ok) {
            displayQAErrorMessage(result.error || "Unexpected error occurred.");
            throw new Error(`Server error: ${response.status}`);
        }

        if (result.generated_text) {
            qaResult.innerHTML = `<h2>Answer</h2><p>${result.generated_text.trim()}</p>`;

            if (result.lime_explanation) {
                renderLimeExplanation('lime-explanation-qa', result.lime_explanation);
            }
        } else {
            displayQAErrorMessage("The server returned an invalid or empty response.");
        }

    } catch (error) {
        console.error(error);
        alert("An error occurred: " + error.message);
    }
}

// ========== Utilities ==========

function resetForm() {
    document.getElementById("prediction-form").reset();
    document.getElementById("prediction-result").style.display = "none";
    document.getElementById("error-message").style.display = "none";
    document.getElementById("lime-explanation-sts").style.display = "none";
    document.getElementById("lime-table-body-sts").innerHTML = '';
}

function resetQAForm() {
    document.getElementById("qa-form").reset();
    document.getElementById("qa-result").style.display = "none";
    document.getElementById("qa-error-message").style.display = "none";
    document.getElementById("lime-explanation-qa").style.display = "none";
    document.getElementById("lime-table-body-qa").innerHTML = '';
}

function displayErrorMessage(message) {
    const errorContainer = document.getElementById("error-message");
    errorContainer.innerText = message;
    errorContainer.style.display = "block";
}

function displayQAErrorMessage(message) {
    const qaErrorContainer = document.getElementById("qa-error-message");
    qaErrorContainer.innerText = message;
    qaErrorContainer.style.display = "block";
}

// Highlight keywords in sentences with LIME results
function highlightTextWithKeywords(sentence, keywords, weights) {
    const wordWeights = {};

    // Create a dictionary: word -> weight
    keywords.forEach((word, index) => {
        wordWeights[word] = weights[index];
    });

    // Sort words by length (longest first to avoid partial matches)
    const sortedWords = Object.keys(wordWeights).sort((a, b) => b.length - a.length);

    let highlighted = sentence;

    sortedWords.forEach(word => {
        const color = getHighlightColor(wordWeights[word]);
        const regex = new RegExp(`\\b${escapeRegex(word)}\\b`, "gi");
        highlighted = highlighted.replace(regex, (match) => `<mark style="background-color: ${color}">${match}</mark>`);
    });

    return highlighted;
}

function escapeRegex(string) {
    // Escape regex special characters
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function getHighlightColor(weight) {
    if (weight > 0) return "rgba(0, 255, 0, 1.00)";   // Positive weight -> Green
    if (weight < 0) return "rgba(255, 0, 0, 1.00)";   // Negative weight -> Red
    return "rgba(128, 128, 128, 1.00)";               // Near zero -> Gray
}

function renderLimeExplanation(containerId, explanation) {
    const limeSection = document.getElementById(containerId);
    const tableBody = document.getElementById(`${containerId.replace('explanation', 'table-body')}`);

    tableBody.innerHTML = '';

    explanation.keywords.forEach((word, index) => {
        const weight = explanation.weights[index];
        const color = getHighlightColor(weight);

        const row = document.createElement('tr');

        const wordCell = document.createElement('td');
        wordCell.style.padding = '8px';
        wordCell.textContent = word;

        const weightCell = document.createElement('td');
        weightCell.style.padding = '8px';
        weightCell.style.textAlign = 'right';
        weightCell.textContent = weight.toFixed(4);

        row.appendChild(wordCell);
        row.appendChild(weightCell);
        tableBody.appendChild(row);
    });

    limeSection.style.display = 'block';
}
