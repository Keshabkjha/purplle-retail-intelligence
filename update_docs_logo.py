import re

with open("app/dashboard.html", "r") as f:
    dashboard_html = f.read()

# Extract the logo block from dashboard.html
match = re.search(r'(<div class="relative w-12 h-12 rounded-xl overflow-hidden[^>]*>\s*<img src="data:image/png;base64,[^"]+"[^>]*>\s*</div>)', dashboard_html)
if match:
    logo_html = match.group(1)
    
    with open("app/docs.html", "r") as f:
        docs_html = f.read()
        
    # Find the PRI text div and replace it
    target_div_pattern = r'<div class="h-11 w-11 rounded-2xl bg-gradient-to-br from-emerald-400 to-sky-400 grid place-items-center text-slate-950 font-black">PRI</div>'
    
    if re.search(target_div_pattern, docs_html):
        docs_html = re.sub(target_div_pattern, logo_html.replace('\\', '\\\\'), docs_html)
        with open("app/docs.html", "w") as f:
            f.write(docs_html)
        print("Successfully updated logo in docs.html")
    else:
        print("Could not find the target PRI div in docs.html")
else:
    print("Could not find the logo block in dashboard.html")
