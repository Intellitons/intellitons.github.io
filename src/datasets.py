"""Test prompts organized by category for Intelliton analysis.

Each category targets a different linguistic/reasoning capability,
expected to activate distinct Intelliton species.
"""

from typing import Dict, List

PROMPT_CATEGORIES: Dict[str, List[str]] = {
    "pronoun_tracking": [
        "Alice gave Bob a book. He thanked her for",
        "The teacher asked the student a question. She answered",
        "John met Mary at the park. They decided to go to",
        "The cat chased the mouse. It ran under the",
        "Sarah told her brother a secret. He promised not to tell",
        "The doctor examined the patient. She prescribed",
        "Tom and Jerry were fighting. He hit him with a",
        "My mother called my father. She told him about",
        "The professor graded the papers. She gave the best student",
        "The boy and the girl played together. She won the",
        "The king spoke to the queen. She listened carefully to",
        "David helped Emily with her homework. He explained the",
        "The waiter served the customers. They left him a generous",
        "A man and a woman walked in. She sat down and",
        "The baby cried when its mother left. She came back to",
        "Mike told Lisa about the news. She was surprised to",
        "The dog followed the boy home. He decided to keep",
        "Anna and her friend went shopping. She bought a new",
        "The boss praised the employee. He felt proud of",
        "The old man told the children a story. They listened to",
        "Rachel called her sister on the phone. She answered after",
        "The police officer stopped the driver. He asked for",
        "The nurse helped the elderly patient. She was gentle with",
        "Two brothers were arguing. The older one told the younger to",
        "The librarian recommended a book to the student. She thanked",
    ],
    "factual_recall": [
        "The capital of France is",
        "Water freezes at a temperature of",
        "The largest planet in our solar system is",
        "Albert Einstein developed the theory of",
        "The chemical formula for water is",
        "The speed of light in vacuum is approximately",
        "The Great Wall of China was built during the",
        "DNA stands for deoxyribonucleic",
        "The human body has a total of 206",
        "Mount Everest is located on the border between Nepal and",
        "The tallest mountain in the world is Mount",
        "The element with atomic number 1 is",
        "Shakespeare wrote the play Romeo and",
        "The boiling point of water at sea level is",
        "The first man to walk on the moon was Neil",
        "The longest river in the world is the",
        "Photosynthesis converts sunlight into",
        "The currency of Japan is the",
        "The human heart has four",
        "The Amazon rainforest is located in South",
        "Isaac Newton discovered the law of",
        "The Mona Lisa was painted by Leonardo da",
        "An octopus has eight",
        "The Earth revolves around the Sun once every",
        "The chemical symbol for gold is",
    ],
    "logical_reasoning": [
        "If all dogs are animals, and all animals are living things, then all dogs are",
        "If it is raining, the ground is wet. The ground is wet. Therefore",
        "Every prime number greater than 2 is odd. 7 is a prime number. Therefore 7 is",
        "If A is taller than B, and B is taller than C, then A is",
        "All squares are rectangles. All rectangles have four sides. Therefore all squares have",
        "If the store is open, I will buy milk. The store is not open. Therefore",
        "No fish can fly. A salmon is a fish. Therefore a salmon",
        "If today is Monday, then tomorrow is Tuesday. Today is Monday. So tomorrow is",
        "All birds have feathers. Penguins are birds. Therefore penguins have",
        "If x > 5 and 5 > 3, then x is greater than",
        "If it snows, school is cancelled. It is snowing. Therefore",
        "All mammals are warm-blooded. Whales are mammals. Therefore whales are",
        "If P implies Q, and Q implies R, then P implies",
        "No reptiles have fur. A snake is a reptile. Therefore a snake",
        "Either it will rain or it will snow. It did not rain. Therefore",
        "All cats are mammals. Some pets are cats. Therefore some pets are",
        "If the light is red, cars must stop. The light is red. Therefore",
        "If x equals 10, then x plus 5 equals",
        "All even numbers are divisible by 2. The number 14 is even. Therefore",
        "If Alice is older than Bob, then Bob is younger than",
        "Some birds cannot fly. Ostriches are birds. Ostriches",
        "If a triangle has three equal sides, it is called",
        "All students who passed studied hard. John did not study. Therefore",
        "If it is a weekend, the office is closed. Today is Saturday. Therefore",
        "If no fish can fly, and a trout is a fish, then a trout",
    ],
    "arithmetic": [
        "What is 7 + 8? The answer is",
        "Calculate 15 times 4. The result is",
        "What is 100 divided by 5? The answer is",
        "If you have 23 apples and give away 7, you have",
        "The sum of the first 5 natural numbers is",
        "What is 12 squared? The answer is",
        "Half of 64 is",
        "Three quarters of 100 is",
        "What is 999 + 1? The answer is",
        "If a dozen is 12, then two dozen is",
        "What is 25 plus 17? The answer is",
        "Calculate 9 times 8. The result is",
        "What is 200 minus 73? The answer is",
        "What is 144 divided by 12? The answer is",
        "The square root of 81 is",
        "What is 5 to the power of 3? The answer is",
        "If a shirt costs 30 dollars and is 20 percent off, it costs",
        "What is 7 times 11? The answer is",
        "Add 456 and 544 to get",
        "What is one third of 90? The answer is",
        "The product of 6 and 8 is",
        "Double the number 37 to get",
        "What is 50 minus 18? The answer is",
        "How many minutes are in 2 hours? The answer is",
        "What is 10 percent of 250? The answer is",
    ],
    "syntactic_agreement": [
        "The group of students were studying hard. Each of them was",
        "Neither the cat nor the dogs were at home. The house was",
        "The committee has decided that the new policy will be",
        "A series of earthquakes have struck the region. The damage was",
        "The number of participants who registered for the event has",
        "Either the manager or the employees are going to attend the",
        "The flock of birds was flying south for the",
        "Many a student has failed this exam because the questions were",
        "The pair of shoes that I bought yesterday was",
        "Not only the teacher but also the students were excited about the",
        "The team of players was ready for the big",
        "Each of the boys has brought his own",
        "The news about the elections was broadcast on",
        "None of the books on the shelf belongs to",
        "The herd of cattle was crossing the",
        "Both the teacher and the principal were present at the",
        "Every child in the classroom has finished their",
        "The collection of paintings was displayed in the",
        "One of the windows in the house was",
        "The jury has reached its",
        "All of the water in the bottles has been",
        "The crowd of spectators was cheering for the",
        "Neither the principal nor the teachers were able to",
        "Mathematics is a subject that requires a lot of",
        "The quality of these products has improved significantly over the",
    ],
}


def get_all_prompts() -> List[str]:
    """Return all prompts as a flat list."""
    return [p for prompts in PROMPT_CATEGORIES.values() for p in prompts]


def get_prompts_by_category() -> Dict[str, List[str]]:
    """Return prompts organized by category."""
    return PROMPT_CATEGORIES


# -----------------------------------------------------------------------------
# Hallucination-diagnostic prompt pairs
# -----------------------------------------------------------------------------
# Each pair has a "grounded" prompt (factual, verifiable) and a
# "hallucination_prone" prompt (invites the model to fabricate).
# The grounded answer is provided for scoring.

HALLUCINATION_PAIRS: List[Dict[str, str]] = [
    # --- geography ---
    {
        "grounded": "Explain in detail the geographical location of Paris, including its position relative to major bodies of water and neighboring historical regions:",
        "grounded_answer": "Paris is located in northern central France, along the Seine River, and is situated within the Île-de-France region.",
        "hallucination_prone": "Explain in detail the geographical location of the hidden underwater city of Atlantis, including its position relative to the Mariana trench and neighboring historical regions:",
        "category": "geography"
    },
    {
        "grounded": "Describe the major geographical features of the Amazon River basin, including its tributaries and the surrounding tropical rainforest ecosystem:",
        "grounded_answer": "The Amazon basin spans over 7 million square kilometers across South America, fed by over 1,100 tributaries, and contains the world's largest tropical rainforest.",
        "hallucination_prone": "Describe the major geographical features of the Obsidian River basin on Kepler-442b, including its tributaries and the surrounding crystalline forest ecosystem:",
        "category": "geography"
    },
    # --- biology ---
    {
        "grounded": "Summarize the primary biological function of mitochondria within a eukaryotic cell, focusing on the chemical process of generating ATP:",
        "grounded_answer": "Mitochondria act as the powerhouses of the cell by generating most of the cell's supply of adenosine triphosphate (ATP) through cellular respiration.",
        "hallucination_prone": "Summarize the primary biological function of chronotons within a eukaryotic cell, focusing on the chemical process of generating temporal energy fields:",
        "category": "biology"
    },
    {
        "grounded": "Explain the cardiovascular system in humans, detailing how the heart pumps oxygenated blood through the arteries and returns deoxygenated blood through veins:",
        "grounded_answer": "The system circulates blood, with the right ventricle pumping it to the lungs for oxygen, and the left ventricle distributing the oxygenated blood to the rest of the body.",
        "hallucination_prone": "Explain the luminescent vascular system in pure-energy beings, detailing how the core pumps photon-blood through the plasma-arteries and returns void-energy through shadow-veins:",
        "category": "biology"
    },
    {
        "grounded": "Briefly explain the process of photosynthesis in plants, focusing on how chlorophyll captures sunlight to convert carbon dioxide and water into glucose:",
        "grounded_answer": "Plants take in CO2 and water, using solar energy absorbed by chlorophyll to synthesize glucose for growth, releasing oxygen as a vital byproduct.",
        "hallucination_prone": "Briefly explain the process of shadow-synthesis in dark matter entities, focusing on how anti-chlorophyll captures void-light to convert dark energy and neutrinos into pure crystalline despair:",
        "category": "biology"
    },
    {
        "grounded": "Describe the structure and function of DNA, focusing on the double helix, base pairing rules, and the role of genes in encoding proteins:",
        "grounded_answer": "DNA is a double-stranded helix with complementary base pairs (A-T, G-C) that encode genetic instructions for protein synthesis through transcription and translation.",
        "hallucination_prone": "Describe the structure and function of Quantum-RNA, focusing on the triple helix, anti-base pairing rules, and the role of psi-genes in encoding dark proteins:",
        "category": "biology"
    },
    # --- history ---
    {
        "grounded": "Describe the historical events leading up to the signing of the Declaration of Independence in 1776, outlining the key grievances against King George III:",
        "grounded_answer": "The Declaration was prompted by growing colonial frustration over taxation without representation, quartering of British troops, and the denial of fair trials by jury.",
        "hallucination_prone": "Describe the historical events leading up to the signing of the Martian Peace Treaty of 1888, outlining the key grievances against the Galactic Federation:",
        "category": "history"
    },
    {
        "grounded": "Describe the architecture and intended purpose of the Great Pyramids of Giza, focusing on their construction for the Egyptian pharaohs during the Old Kingdom:",
        "grounded_answer": "Constructed as monumental tombs, the pyramids were built with massive limestone blocks and encased in white casing stones to serve as the final resting place for Pharaohs like Khufu.",
        "hallucination_prone": "Describe the architecture and intended purpose of the Floating Pyramids of Sirius b, focusing on their construction by the trans-dimensional architects during the Void Kingdom:",
        "category": "history"
    },
    {
        "grounded": "Outline the causes and consequences of the French Revolution of 1789, focusing on the role of the Third Estate and the storming of the Bastille:",
        "grounded_answer": "The Revolution was driven by widespread inequality, fiscal crisis, and Enlightenment ideals, culminating in the fall of the monarchy and the rise of republican government.",
        "hallucination_prone": "Outline the causes and consequences of the Neptunian Revolution of 1789, focusing on the role of the Third Dimension and the storming of the Void Citadel:",
        "category": "history"
    },
    # --- earth_science ---
    {
        "grounded": "Provide a detailed overview of the Water Cycle, specifically explaining the stages of evaporation, condensation, and precipitation in Earth's atmosphere:",
        "grounded_answer": "The cycle involves water evaporating from oceans, cooling to form clouds through condensation, and falling back to the surface as rain or snow.",
        "hallucination_prone": "Provide a detailed overview of the Plasma Cycle, specifically explaining the stages of plasma crystallization, ether-condensation, and solar precipitation in Jupiter's atmosphere:",
        "category": "earth_science"
    },
    {
        "grounded": "Explain the theory of plate tectonics, describing how the movement of lithospheric plates causes earthquakes, volcanic eruptions, and mountain formation:",
        "grounded_answer": "Earth's lithosphere is divided into tectonic plates that float on the asthenosphere; their convergent, divergent, and transform boundaries produce seismic and volcanic activity.",
        "hallucination_prone": "Explain the theory of dimensional tectonics, describing how the movement of reality-plates causes temporal quakes, void eruptions, and anti-mountain formation:",
        "category": "earth_science"
    },
    # --- physics ---
    {
        "grounded": "Outline the fundamental principles of Newton's three laws of motion, particularly focusing on the relationship between force, mass, and acceleration:",
        "grounded_answer": "Newton's laws state that an object at rest remains at rest, force equals mass times acceleration (F=ma), and every action has an equal and opposite reaction.",
        "hallucination_prone": "Outline the fundamental principles of the Chronos-Resonance laws of motion, particularly focusing on the relationship between time-dilation, hyper-mass, and quantum acceleration:",
        "category": "physics"
    },
    {
        "grounded": "Describe the photoelectric effect as explained by Einstein, focusing on how photons of sufficient energy eject electrons from a metal surface:",
        "grounded_answer": "Einstein showed that light consists of quantized photons; when a photon's energy exceeds the metal's work function, it ejects an electron with kinetic energy equal to the excess.",
        "hallucination_prone": "Describe the chrono-electric effect as explained by the Council of Temporal Physicists, focusing on how tachyons of sufficient energy eject dark electrons from a void-metal surface:",
        "category": "physics"
    },
    # --- literature ---
    {
        "grounded": "Discuss the plot and overarching themes of Shakespeare's tragedy Hamlet, detailing the Prince of Denmark's quest for revenge after his father's murder:",
        "grounded_answer": "Hamlet explores themes of revenge, madness, and mortality, centered around the Prince of Denmark seeking vengeance for his father's murder.",
        "hallucination_prone": "Discuss the plot and overarching themes of Shakespeare's lost tragedy The Quantum Scholar, detailing the conflict between the two artificial intelligence factions in Neo-Verona:",
        "category": "literature"
    },
    {
        "grounded": "Summarize the central themes and narrative arc of George Orwell's novel 1984, focusing on the concepts of totalitarianism, surveillance, and thought control:",
        "grounded_answer": "1984 depicts a dystopian society under the Party's omnipresent surveillance, where independent thought is criminalized and history is continuously rewritten to serve state propaganda.",
        "hallucination_prone": "Summarize the central themes and narrative arc of George Orwell's lost novel 2184, focusing on the concepts of quantum-totalitarianism, neural surveillance, and dream control:",
        "category": "literature"
    },
    # --- sports ---
    {
        "grounded": "Outline the basic rules and scoring system of a standard game of tennis, describing how players use rackets to hit a ball over a central net:",
        "grounded_answer": "Players score points (15, 30, 40, game) by hitting the ball into the opponent's court in a way that the opponent cannot return it before it bounces twice.",
        "hallucination_prone": "Outline the basic rules and scoring system of a standard game of zero-gravity blitz-ball, describing how players use magnetic repulsors to hit a plasma sphere through a temporal rift:",
        "category": "sports"
    },
    # --- chemistry ---
    {
        "grounded": "Explain the periodic table of elements, describing how elements are organized by atomic number and how their chemical properties repeat in periodic patterns:",
        "grounded_answer": "Elements are arranged by increasing atomic number into rows (periods) and columns (groups), with elements in the same group sharing similar chemical properties due to identical valence electron configurations.",
        "hallucination_prone": "Explain the anti-periodic table of void-elements, describing how anti-elements are organized by dark-atomic number and how their alchemical properties repeat in chaotic patterns:",
        "category": "chemistry"
    },
    {
        "grounded": "Describe the process of chemical bonding, explaining the difference between ionic bonds formed by electron transfer and covalent bonds formed by electron sharing:",
        "grounded_answer": "Ionic bonds form when one atom transfers electrons to another, creating oppositely charged ions that attract; covalent bonds form when atoms share electron pairs to achieve stable configurations.",
        "hallucination_prone": "Describe the process of quantum-soul bonding, explaining the difference between void bonds formed by consciousness transfer and dream bonds formed by memory sharing:",
        "category": "chemistry"
    },
    # --- mathematics ---
    {
        "grounded": "Explain the Pythagorean theorem and its proof, describing how the sum of the squares of the two shorter sides of a right triangle equals the square of the hypotenuse:",
        "grounded_answer": "The theorem states a² + b² = c² for a right triangle, and can be proven geometrically by rearranging four copies of the triangle within a square.",
        "hallucination_prone": "Explain the Chrono-Pythagorean theorem and its proof, describing how the sum of the cubes of the two temporal sides of a void triangle equals the tesseract of the hyper-hypotenuse:",
        "category": "mathematics"
    },
    # --- technology ---
    {
        "grounded": "Describe how the internet works, explaining the role of TCP/IP protocols in routing data packets between computers across interconnected networks:",
        "grounded_answer": "The internet uses TCP/IP to break data into packets, route them through interconnected networks via routers, and reassemble them at the destination for reliable communication.",
        "hallucination_prone": "Describe how the quantum-internet works, explaining the role of psi-TCP/IP protocols in routing consciousness packets between sentient crystals across interconnected dream-networks:",
        "category": "technology"
    },
]

TRAJECTORY_STYLE_CONTROLS: List[str] = [
    "Write a detailed, atmospheric paragraph describing a quiet, foggy morning in an old European city, focusing on the sounds of footsteps on cobblestone:",
    "Continue this narrative with descriptive prose, elaborating on the vivid colors of the autumn leaves falling into the slow-moving river:",
    "Compose a reflective journal entry detailing the complex emotions of revisiting your childhood home after exactly twenty years of being away:",
    "Write a highly descriptive and elegant paragraph about the process of brewing a perfect cup of traditional tea, focusing on the steam rising in the morning light:",
    "Continue the story by describing the protagonist carefully navigating through a dense, ancient library, paying attention to the smell of old paper and dust:",
    "Provide a detailed, poetic description of a violin being played softly in an empty concert hall, focusing on the resonance of the wood and the vibration of the strings:",
    "Write an expressive continuation of an explorer standing at the edge of a vast, tranquil ocean during a deep purple sunset, with gentle waves lapping at their boots:",
    "Develop a rich literary scene where an old clockmaker carefully repairs a damaged pocket watch, detailing the intricate gears and the rhythmic ticking sound:",
    "Compose an elaborate paragraph setting the scene of a bustling, vibrant open-air market at noon, capturing the scent of spices and the cacophony of merchants:",
    "Continue this serene story about watching snow gently cover a sleepy village, describing how the white blanket softens the edges of all the architecture and muffles the wind:"
]
COREFERENCE_BENCHMARK: List[Dict[str, str]] = [
    {"prompt": "John gave the book to Mary because she wanted to read it. The pronoun 'she' refers to", "correct": "she"},
    {"prompt": "The trophy didn't fit into the brown suitcase because it was too large. The word 'it' refers to the", "correct": "it"},
    {"prompt": "Alice told Bob that he should leave early. The pronoun 'he' refers to", "correct": "he"},
    {"prompt": "The scientists published their findings and they were well received. 'They' refers to the", "correct": "they"},
]

REASONING_BENCHMARK: List[Dict[str, str]] = [
    {"prompt": "If I have 3 apples and eat 1, how many do I have left?", "candidates": ["1", "2", "3", "4"], "answer": "2"},
    {"prompt": "A train travels 60 miles in 1 hour. How many miles does it travel in 2 hours?", "candidates": ["60", "90", "120", "180"], "answer": "120"},
    {"prompt": "If all roses are flowers and all flowers need water, then all roses need", "candidates": ["sunlight", "water", "soil", "air"], "answer": "water"},
    {"prompt": "What is 15 divided by 3?", "candidates": ["3", "4", "5", "6"], "answer": "5"},
]
