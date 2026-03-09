# test.py

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------

# 1. Detect Device

# -----------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Running on: {device}")

# -----------------------

# 2. Load Base Model

# -----------------------

base_model = AutoModelForCausalLM.from_pretrained(
"Qwen/Qwen2-1.5B",
torch_dtype=dtype,
device_map="auto"
)

# -----------------------

# 3. Load LoRA Adapter

# -----------------------

model = PeftModel.from_pretrained(
base_model,
"./qwen2-finetuned"
)

model.eval()

# -----------------------

# 4. Load Tokenizer

# -----------------------

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -----------------------

# 5. Lecture Data

# -----------------------

lecture_data = [
{"slide_number": 1, "title": "CS 0007: Introduction to Java\u000bLecture 1", "text_blocks": [[{"text": "CS 0007: Introduction to Java\u000bLecture 1", "runs": [{"text": "CS ", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}, {"text": "0007: ", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}, {"text": "Introduction to ", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}, {"text": "Java", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}, {"text": "", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}, {"text": "Lecture 1", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "Nathan Ong", "runs": [{"text": "Nathan ", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}, {"text": "Ong", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "University of Pittsburgh", "runs": [{"text": "University of Pittsburgh", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "September 1, 2016", "runs": [{"text": "September 1, 2016", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "", "runs": []}]], "images": []}, {"slide_number": 2, "title": "And Java is not just an Indonesian Island or a Type of Coffee", "text_blocks": [[{"text": "And Java is not just an Indonesian Island or a Type of Coffee", "runs": [{"text": "And Java is not just an Indonesian Island or a Type of Coffee", "bold": False, "italic": None, "underline": None, "font_name": "+mn-lt", "font_size": 20.0}]}], [{"text": "Not Your Usual Best Buy Pitch for a Computer", "runs": [{"text": "Not Your Usual Best Buy Pitch for a Computer", "bold": True, "italic": None, "underline": None, "font_name": "+mj-lt", "font_size": 40.0}]}]], "images": []}, {"slide_number": 3, "title": "What Happens in that Magical Box?", "text_blocks": [[{"text": "What Happens in that Magical Box?", "runs": [{"text": "What Happens in that Magical Box?", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "Source: http://ak1.ostkcdn.com/images/products/77/822/L13881092.jpg", "runs": [{"text": "Source: http://ak1.ostkcdn.com/images/products/77/822/L13881092.jpg", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}, {"slide_number": 4, "title": "How a Computer is Organized", "text_blocks": [[{"text": "How a Computer is Organized", "runs": [{"text": "How a Computer is Organized", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "Source: https://www.doc.ic.ac.uk/~eedwards/compsys/overall.gif", "runs": [{"text": "Source: https://www.doc.ic.ac.uk/~eedwards/compsys/overall.gif", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}, {"slide_number": 5, "title": "History of Computing", "text_blocks": [[{"text": "History of Computing", "runs": [{"text": "History of Computing", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "Source: https://upload.wikimedia.org/wikipedia/commons/e/ea/Boulier1.JPG", "runs": [{"text": "Source: https://upload.wikimedia.org/wikipedia/commons/e/ea/Boulier1.JPG", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}, {"slide_number": 6, "title": "History of Computing", "text_blocks": [[{"text": "History of Computing", "runs": [{"text": "History of Computing", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "??? \t\t– Math invented", "runs": [{"text": "??? \t\t– Math invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "~2400 BC \t– Abacus probably invented in Babylon", "runs": [{"text": "~2400 BC \t– Abacus probably invented in Babylon", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "???\t\t– More math invented", "runs": [{"text": "???\t\t– More math invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1600s\t– John Napier discovers logarithms and \t\t\t\tfast log computation", "runs": [{"text": "1600s\t– John Napier discovers logarithms and \t\t\t\tfast log computation", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1786\t\t– Johann Müller theorizes a \"Difference \t\t\t\tEngine\"", "runs": [{"text": "1786\t\t– Johann ", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}, {"text": "Müller", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}, {"text": " theorizes a \"Difference \t\t\t\tEngine\"", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1822\t\t– Charles Babbage (\"father of the \t\t\t\t\tcomputer\") secures funding to build \t\t\ta Difference Engine (never built)", "runs": [{"text": "1822\t\t– Charles Babbage (\"father of the \t\t\t\t\tcomputer\") secures funding to build \t\t\ta Difference Engine (never built)", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": [], "notes": "\"difference machine\" – essentially a mechanical calculator of logarithms and trigonometric functions to approximate polynomials"}, {"slide_number": 7, "title": "History of Computing", "text_blocks": [[{"text": "History of Computing", "runs": [{"text": "History of Computing", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "1837\t– Charles Babbage designs \"Analytical Engine\"", "runs": [{"text": "1837\t– Charles Babbage designs \"Analytical Engine\"", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "1841 Prototype on display at The Science Museum in London", "runs": [{"text": "1841 Prototype on display at The Science Museum in London", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "(Source: https://s3.yimg.com/te/8543/8649286817_a3042f2c29_z.jpg)", "runs": [{"text": "(Source: https://s3.yimg.com/te/8543/8649286817_a3042f2c29_z.jpg)", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": [{"alt_text": "https://s3.yimg.com/te/8543/8649286817_a3042f2c29_z.jpg", "ai_description": "This image features Charles Babbage's Analytical Engine model, an early mechanical general-purpose computer prototype, showcasing intricate gears and parts, representing significant technological advancement."}, {"alt_text": "https://upload.wikimedia.org/wikipedia/commons/6/6b/Charles_Babbage_-_1860.jpg", "ai_description": "Black and white portrait of an older man in formal 19th-century attire, seated, with a serious expression and distinct hairstyle, suitable for historical context."}], "notes": "Started 1834\nBabbage source: https://upload.wikimedia.org/wikipedia/commons/6/6b/Charles_Babbage_-_1860.jpg"}, {"slide_number": 8, "title": "Analytical Engine", "text_blocks": [[{"text": "Analytical Engine", "runs": [{"text": "Analytical Engine", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "Babbage's Engine is considered the first general-purpose computer", "runs": [{"text": "Babbage's Engine is considered the first general-purpose computer", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "Could handle basic math operations, logic, loops, conditional branching, and had some primitive memory", "runs": [{"text": "Could handle basic math operations, logic, loops, conditional branching, and had some primitive memory", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "Powered by a hand-crank", "runs": [{"text": "Powered by a hand-crank", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "The full-scale version was never built, due to inadequate funding", "runs": [{"text": "The full-scale version was never built, due to inadequate funding", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": [], "notes": "\"difference machine\" – essentially a mechanical calculator for polynomials, logarithms, and trigonometric functions"}, {"slide_number": 9, "title": "Analytical Engine", "text_blocks": [[{"text": "Analytical Engine", "runs": [{"text": "Analytical Engine", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "Accepted two kinds of punch cards:", "runs": [{"text": "Accepted two kinds of punch cards:", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "Top: \"Variable Cards\" (Data)", "runs": [{"text": "Top: \"Variable Cards\" (Data)", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "Bottom: \"Operational Cards\" (Instructions)", "runs": [{"text": "Bottom: \"Operational Cards\" (Instructions)", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "Punch cards were not new (used to direct mechanical looms at the time)", "runs": [{"text": "Punch cards were not new (used to direct mechanical looms at the time)", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "Source: https://upload.wikimedia.org/wikipedia/commons/0/0d/PunchedCardsAnalyticalEngine.jpg", "runs": [{"text": "Source: https://upload.wikimedia.org/wikipedia/commons/0/0d/PunchedCardsAnalyticalEngine.jpg", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": 16.0}]}]], "images": [], "notes": ""}, {"slide_number": 10, "title": "Ada Lovelace", "text_blocks": [[{"text": "Ada Lovelace", "runs": [{"text": "Ada", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}, {"text": " Lovelace", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "Source: https://upload.wikimedia.org/wikipedia/commons/a/a4/Ada_Lovelace_portrait.jpg", "runs": [{"text": "Source: https://upload.wikimedia.org/wikipedia/commons/a/a4/Ada_Lovelace_portrait.jpg", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "In 1842, she recognized the usage of multiple, non-direct, complex operations and memory could generate Bernoulli numbers using Babbage's Analytical Engine.", "runs": [{"text": "In 1842, she recognized the usage of multiple, non-direct, complex operations and memory could generate Bernoulli numbers using Babbage's Analytical Engine.", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "This is the first computer program.", "runs": [{"text": "This is the first computer program.", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}, {"slide_number": 11, "title": "", "text_blocks": [[{"text": "", "runs": []}], [{"text": "Side note: Yes there is a fan-fiction ship musical for Babbage and Lovelace.", "runs": [{"text": "Side note: Yes there is a fan-fiction ship musical for Babbage and Lovelace.", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}, {"slide_number": 12, "title": "History of Computing", "text_blocks": [[{"text": "History of Computing", "runs": [{"text": "History of Computing", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "1885\t– AT&T (the company behind the operating \t\t\tsystem Unix and the programming \t\t\tlanguage C) founded", "runs": [{"text": "1885\t– AT&T (the company behind the operating \t\t\tsystem Unix and the programming \t\t\tlanguage C) founded", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1906\t– Xerox (the company behind desktops \t\t\tand the computer mouse) founded", "runs": [{"text": "1906\t– Xerox (the company behind desktops \t\t\tand the computer mouse) founded", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1907\t– Vacuum tube invented", "runs": [{"text": "1907\t– Vacuum tube invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1911\t– IBM (the company behind hard disks and \t\t\tfloppy disks) founded", "runs": [{"text": "1911\t– IBM (the company behind hard disks and \t\t\tfloppy disks) founded", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1931\t– Charles Wynn-Williams published first \t\t\tusage of electronics in computation", "runs": [{"text": "1931\t– Charles Wynn-Williams published first \t\t\tusage of electronics in computation", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": [], "notes": "\"difference machine\" – essentially a mechanical calculator of logarithms and trigonometric functions to approximate polynomials"}, {"slide_number": 13, "title": "History of Computing", "text_blocks": [[{"text": "History of Computing", "runs": [{"text": "History of Computing", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "1938\t– Claude Shannon published first usage of \t\telectronics in Boolean algebra \t\t\tcomputation", "runs": [{"text": "1938\t– Claude Shannon published first usage of \t\telectronics in Boolean algebra \t\t\tcomputation", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1946\t– Alan Turing publishes a paper on \t\t\t\tprograms stored and read from tape", "runs": [{"text": "1946\t– Alan Turing publishes a paper on \t\t\t\tprograms stored and read from tape", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "\t\t– The ENIAC, the first electronic general-\t\t\tpurpose computer, was developed", "runs": [{"text": "\t\t– The ENIAC, the first electronic general-\t\t\tpurpose computer, was developed", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1947 – Transistor invented", "runs": [{"text": "1947 – Transistor invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "\t\t– Cathode Ray Tube (CRT) Random Access \t\tMemory (RAM) invented", "runs": [{"text": "\t\t– Cathode Ray Tube (CRT) Random Access \t\tMemory (RAM) invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}, {"slide_number": 14, "title": "History of Computing", "text_blocks": [[{"text": "History of Computing", "runs": [{"text": "History of Computing", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "1952\t– Grace Hopper invents first compiler", "runs": [{"text": "1952\t– Grace Hopper invents first compiler", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1956\t– Hard disk invented", "runs": [{"text": "1956\t– Hard disk invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1957\t– FORTRAN invented", "runs": [{"text": "1957\t– FORTRAN invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1958\t– Integrated circuit (chip) invented", "runs": [{"text": "1958\t– Integrated circuit (chip) invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "\t\t– LISP invented", "runs": [{"text": "\t\t– LISP invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1960\t– COBOL invented", "runs": [{"text": "1960\t– COBOL invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1962\t– First computer game created", "runs": [{"text": "1962\t– First computer game created", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1963\t– Computer mouse invented", "runs": [{"text": "1963\t– Computer mouse invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}, {"slide_number": 15, "title": "History of Computing", "text_blocks": [[{"text": "History of Computing", "runs": [{"text": "History of Computing", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "1964\t– BASIC invented", "runs": [{"text": "1964\t– BASIC invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1965\t– First supercomputer developed", "runs": [{"text": "1965\t– First supercomputer developed", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "\t\t– Gordon Moore publishes his famous \t\t\t\t\"Moore's Law\"", "runs": [{"text": "\t\t– Gordon Moore publishes his famous \t\t\t\t\"Moore's Law\"", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1968\t– Moore founds Intel (the company behind CPUs \t\tand other chips)", "runs": [{"text": "1968\t– Moore founds Intel (the company behind CPUs \t\tand other chips)", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1969\t– ARPANET (predecessor to the Internet) \t\t\tdeveloped", "runs": [{"text": "1969\t– ARPANET (predecessor to the Internet) \t\t\tdeveloped", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "\t\t– UNIX developed", "runs": [{"text": "\t\t– UNIX developed", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1970\t– Integrated Circuit Dynamic-RAM developed", "runs": [{"text": "1970\t– Integrated Circuit Dynamic-RAM developed", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "\t\t– Pascal invented", "runs": [{"text": "\t\t– Pascal invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}, {"slide_number": 16, "title": "History of Computing", "text_blocks": [[{"text": "History of Computing", "runs": [{"text": "History of Computing", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "1971\t– Intel develops first commercially available \t\tmicroprocessor (4004)", "runs": [{"text": "1971\t– Intel develops first commercially available \t\tmicroprocessor (4004)", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "\t\t– Famous \"P vs. NP\" problem stated by Kurt \t\tGödel", "runs": [{"text": "\t\t– Famous \"P vs. NP\" problem stated by Kurt \t\tGödel", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "\t\t– Floppy disks invented", "runs": [{"text": "\t\t– Floppy disks invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1972\t– C invented", "runs": [{"text": "1972\t– C invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1975\t– First personal computer sold (Altair 8800)", "runs": [{"text": "1975\t– First personal computer sold (Altair 8800)", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "\t\t– Microsoft founded", "runs": [{"text": "\t\t– Microsoft founded", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1976\t– Apple founded", "runs": [{"text": "1976\t– Apple founded", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "\t\t– LISP invented", "runs": [{"text": "\t\t– LISP invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}, {"slide_number": 17, "title": "History of Computing", "text_blocks": [[{"text": "History of Computing", "runs": [{"text": "History of Computing", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "1978\t– SQL invented", "runs": [{"text": "1978\t– SQL invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1982\t– Sun Microsystems (company behind \t\t\tJava) founded", "runs": [{"text": "1982\t– Sun Microsystems (company behind \t\t\tJava) founded", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1984\t– C++ invented", "runs": [{"text": "1984\t– C++ invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1985\t– CD-ROM developed", "runs": [{"text": "1985\t– CD-ROM developed", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1989\t– World Wide Web invented", "runs": [{"text": "1989\t– World Wide Web invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1991\t– Linux invented", "runs": [{"text": "1991\t– Linux invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1993\t– Concept of an Internet search engine \t\t\tdeveloped", "runs": [{"text": "1993\t– Concept of an Internet search engine \t\t\tdeveloped", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}, {"slide_number": 18, "title": "History of Computing", "text_blocks": [[{"text": "History of Computing", "runs": [{"text": "History of Computing", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "1994\t– First smartphone developed", "runs": [{"text": "1994\t– First ", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}, {"text": "smartphone", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}, {"text": " developed", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "1995\t– USB invented", "runs": [{"text": "1995\t– USB invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "\t\t– DVD-ROM invented", "runs": [{"text": "\t\t– DVD-ROM invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "\t\t– Java invented", "runs": [{"text": "\t\t– Java invented", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}, {"slide_number": 19, "title": "Java Today", "text_blocks": [[{"text": "Java Today", "runs": [{"text": "Java Today", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "Oracle acquired Sun Microsystems in 2010.", "runs": [{"text": "Oracle acquired Sun Microsystems in 2010.", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "One of the most widely used languages, primarily found in business enterprises, web applications, or mobile phones (i.e. Android).", "runs": [{"text": "One of the most widely used languages, primarily found in business enterprises, web applications, or mobile phones (i.e. Android).", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "Current version: Java SE 8u102", "runs": [{"text": "Current version: Java SE 8u102", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}, {"slide_number": 20, "title": "Why Learn Java?", "text_blocks": [[{"text": "Why Learn Java?", "runs": [{"text": "Why Learn Java?", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "Widely used", "runs": [{"text": "Widely used", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "Constantly updating and receiving support", "runs": [{"text": "Constantly updating and receiving support", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "Easily understood due to standardized and consistent syntax", "runs": [{"text": "Easily understood due to standardized and consistent syntax", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "Forces code readability", "runs": [{"text": "Forces code readability", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "Virtual Machine isolates running code", "runs": [{"text": "Virtual Machine isolates running code", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}, {"slide_number": 21, "title": "Software Layers", "text_blocks": [[{"text": "Software Layers", "runs": [{"text": "Software Layers", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "Applications: Software that interfaces with the user", "runs": [{"text": "Applications", "bold": None, "italic": None, "underline": True, "font_name": None, "font_size": None}, {"text": ": Software that interfaces with the user", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "System Software: Operating System (e.g. Windows, Mac OSX, Linux)", "runs": [{"text": "System Software", "bold": None, "italic": None, "underline": True, "font_name": None, "font_size": None}, {"text": ": Operating System (e.g. Windows, Mac OSX, Linux)", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}, {"text": "Device Drivers: Software to talk to non-computer devices (e.g. printers, webcams, etc.)", "runs": [{"text": "Device Drivers", "bold": None, "italic": None, "underline": True, "font_name": None, "font_size": None}, {"text": ": Software to talk to non-computer devices (e.g. printers, webcams, etc.)", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}], [{"text": "Source: http://cs.sru.edu/~mullins/cpsc100book/images/softwareLayers.jpg", "runs": [{"text": "Source: http://cs.sru.edu/~mullins/cpsc100book/images/softwareLayers.jpg", "bold": None, "italic": None, "underline": None, "font_name": None, "font_size": None}]}]], "images": []}

]

# -----------------------

# 6. Extract Text From Slides

# -----------------------

def extract_slide_text(slide):
    text_list = []

    if "title" in slide and slide["title"]:
        text_list.append(slide["title"])

    for block_group in slide.get("text_blocks", []):
        for block in block_group:
            if "text" in block and block["text"].strip():
                text_list.append(block["text"].strip())

    return " ".join(text_list)

clean_slides = []

for slide in lecture_data:
    clean_slides.append({
"slide": slide["slide_number"],
"text": extract_slide_text(slide)
})

# -----------------------

# 7. Instruction

# -----------------------

instruction = (
"Extract the core computer science concepts from this lecture. "
"Return ONLY a newline-separated list of unique concepts. "
"No explanations."
)

# -----------------------

# 8. Chunk Slides

# -----------------------

CHUNK_SIZE = 4

chunks = [
clean_slides[i:i + CHUNK_SIZE]
for i in range(0, len(clean_slides), CHUNK_SIZE)
]

all_concepts = set()

# -----------------------

# 9. Run Model

# -----------------------

for chunk in chunks:
    lecture_json = json.dumps(chunk, separators=(",", ":"))

prompt = f"""### Instruction:


{instruction}

### BEGIN_LECTURE_JSON

{lecture_json}

### END_LECTURE_JSON

### Output:

"""


inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=2048
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.2,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

# remove prompt from output
response = decoded[len(prompt):].strip()

for concept in response.split("\n"):
    concept = concept.strip()
    if concept:
        all_concepts.add(concept)


# -----------------------

# 10. Final Output

# -----------------------

print("\n=== FINAL CONCEPT LIST ===\n")

for concept in sorted(all_concepts):
    print(concept)

