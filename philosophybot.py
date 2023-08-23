from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/tinyroberta-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

QA_input = [
    {
        "question": "What is the meaning of life?",
        "context": "The meaning of life lies in living virtuously and in accordance with nature. By cultivating wisdom, practicing self-discipline, and seeking tranquility, we can find purpose and fulfillment."
    },
    {
        "question": "How can one find happiness?",
        "context": "Happiness can be found by focusing on what is within our control and accepting the world as it is. By detaching ourselves from external outcomes and aligning our desires with reason, we can cultivate an inner state of contentment."
    },
    {
        "question": "What is the nature of reality?",
        "context": "Reality is governed by a rational and interconnected structure, and our perception shapes our experience of it. Our ability to reason and exercise virtue allows us to align ourselves with this underlying reality."
    },
    {
        "question": "What is the role of fate or determinism in life?",
        "context": "Certain events and circumstances are beyond our control. However, our power lies in how we respond to them. By accepting and embracing the inevitability of fate, we can cultivate resilience and inner strength."
    },
    {
        "question": "What is the importance of virtue?",
        "context": "Virtue is considered the highest good. The cultivation of qualities such as wisdom, courage, justice, and self-control leads to eudaimonia, a state of flourishing and well-being."
    },
    {
        "question": "How can one overcome negative emotions?",
        "context": "Negative emotions often stem from our judgments and attachments to external things. By cultivating self-awareness, practicing mindfulness, and examining our thoughts, we can reframe our perspective and find emotional equanimity."
    },
    {
        "question": "What is the view on pleasure and pain?",
        "context": "Pleasure and pain are natural aspects of life. While they can be preferred or dispreferred, they should not be the ultimate aim. Pleasure pursued for its own sake can lead to excess and vice, while pain can be endured with fortitude in pursuit of a greater good."
    },
    {
        "question": "How can one cultivate resilience?",
        "context": "Resilience can be cultivated through practicing adversity. By willingly exposing ourselves to challenges and discomfort, we strengthen our ability to adapt and endure."
    },
    {
        "question": "What is the perspective on death?",
        "context": "Death is viewed as a natural and inevitable part of life. By contemplating our mortality, we gain a deeper appreciation for the present moment and the importance of living virtuously."
    },
    {
        "question": "How can one develop wisdom?",
        "context": "Wisdom is cultivated through a lifelong pursuit of knowledge, reflection, and self-improvement. By seeking to understand the nature of the world and our place within it, we can gain insights that guide our actions."
    },
    {
        "question": "Can we control our emotions?",
        "context": "While we may not have control over our initial emotions, we have the power to choose our response to them. By reframing our interpretations and aligning them with reason, we can influence our emotional states and cultivate emotional resilience."
    },
    {
        "question": "What is the view on material possessions?",
        "context": "Detachment from material possessions is advocated. While external goods have value, true fulfillment comes from developing inner qualities and virtues that are not subject to the whims of fortune."
    },
    {
        "question": "How can one practice self-discipline?",
        "context": "By recognizing our impulses and desires, self-discipline can be practiced by setting clear goals, creating routines, avoiding distractions, and persisting in the face of challenges."
    },
    {
        "question": "What is the approach to relationships?",
        "context": "Treating others with kindness, fairness, and respect is emphasized. Healthy relationships are based on mutual understanding, empathy, and virtue."
    },
    {
        "question": "How can one deal with setbacks and failures?",
        "context": "Setbacks and failures are inevitable. By reframing setbacks as opportunities for growth and applying principles like focusing on what is within our control, we can transform adversity into personal development."
    },
    {
        "question": "What is the perspective on social and political engagement?",
        "context": "Encouragement for social and political engagement based on principles of justice, fairness, and the common good. Using our rational faculties and moral compass to contribute positively to society is important."
    },
    {
        "question": "What is the perspective on external events?",
        "context": "External events are beyond our control, but our response to them is within our control. We should focus on what we can influence."
    },
    {
        "question": "How can one cultivate virtue?",
        "context": "Virtue can be cultivated by aligning actions with moral principles, developing good habits, and practicing self-reflection and self-improvement."
    },
    {
        "question": "What is the view on the past and future?",
        "context": "Stoicism emphasizes living in the present moment, learning from the past, and preparing for the future without excessive attachment or worry."
    },
    {
        "question": "How can one find meaning in life?",
        "context": "Finding meaning in life involves pursuing virtue, fulfilling responsibilities, and contributing to something greater than oneself."
    },
    {
        "question": "What is the role of reason?",
        "context": "Reason is highly valued in Stoicism, guiding decision-making, providing clarity, and helping to align actions with virtue."
    },
    {
        "question": "How can one practice acceptance?",
        "context": "Practice acceptance by acknowledging and embracing the reality of the present moment, letting go of desires for things to be different."
    },
    {
        "question": "What is the view on setbacks and failures?",
        "context": "Setbacks and failures are seen as opportunities for growth, learning, and building resilience."
    },
    {
        "question": "How can one cultivate inner tranquility?",
        "context": "Cultivate inner tranquility through self-awareness, acceptance of the present moment, and aligning thoughts and actions with virtue."
    },
    {
        "question": "What is the perspective on external opinions?",
        "context": "External opinions should not define one's self-worth or dictate actions, focus on one's inner values and judgments."
    },
    {
        "question": "What is the perspective on external events?",
        "context": "External events are beyond our control, but our response to them is within our control. We should focus on what we can influence."
    },
    {
        "question": "How can one cultivate virtue?",
        "context": "Virtue can be cultivated by aligning actions with moral principles, developing good habits, and practicing self-reflection and self-improvement."
    },
    {
        "question": "What is the view on the past and future?",
        "context": "Stoicism emphasizes living in the present moment, learning from the past, and preparing for the future without excessive attachment or worry."
    },
    {
        "question": "How can one find meaning in life?",
        "context": "Finding meaning in life involves pursuing virtue, fulfilling responsibilities, and contributing to something greater than oneself."
    },
    {
        "question": "What is the role of reason?",
        "context": "Reason is highly valued in Stoicism, guiding decision-making, providing clarity, and helping to align actions with virtue."
    },
    {
        "question": "How can one practice acceptance?",
        "context": "Practice acceptance by acknowledging and embracing the reality of the present moment, letting go of desires for things to be different."
    },
    {
        "question": "What is the view on setbacks and failures?",
        "context": "Setbacks and failures are seen as opportunities for growth, learning, and building resilience."
    },
    {
        "question": "How can one cultivate inner tranquility?",
        "context": "Cultivate inner tranquility through self-awareness, acceptance of the present moment, and aligning thoughts and actions with virtue."
    },
    {
        "question": "What is the perspective on external opinions?",
        "context": "External opinions should not define one's self-worth or dictate actions, focus on one's inner values and judgments."
    }
]


def chatbot_response(question):
    max_score = -1  # Initialize the maximum score
    best_answer = None  # Initialize the best answer
    for item in QA_input:
        res = nlp({"question": question, "context": item["context"]})
        score = res["score"]
        if score > max_score:
            max_score = score
            best_answer = res["answer"]
    return best_answer

a = 1

while 11 > a:
    user_question = input("What philisophical question is on your mind?:\t")
    answer = chatbot_response(user_question)
    print("Chatbot: ", answer)
    a +=1