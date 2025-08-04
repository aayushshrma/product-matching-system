from fastapi import FastAPI, File, Form, UploadFile
from inference import process_image
from inference import process_text
from vector_db import search_nearest_embeddings
from metadata_db import get_product_metadata
from logs_db import db
from logs_db import log_query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse


app = FastAPI()
app.mount("/catalog", StaticFiles(directory="catalog"), name='catalog')

@app.post("/match")
async def match_product(file: UploadFile = File(), name: str = Form()):
    image_bytes = await file.read()
    try:
        image_emb = await process_image(image_bytes=image_bytes)
        text_emb = await process_text(input_text=name)
        nearest_match = await search_nearest_embeddings(image_emb=image_emb, text_emb=text_emb)
        top_match_id = nearest_match[0]["product_id"]
        print(f"top match id: {top_match_id}")
        match = await get_product_metadata(top_match_id)
        print(f"metadata: {match}")
        html = f"""
                <html>
                    <head><title>Product Match Result</title></head>
                    <body>
                        <h2>Top Matching Product</h2>
                        <p><b>Name:</b> {match['name']}</p>
                        <p><b>Price:</b> {match['price']}</p>
                        <p><b>Category:</b> {match['category']}</p>
                        <img src="/{match['image_url']}" alt="{match['name']}" width="300"/>
                        <br><br>
                        <a href="/">Try another</a>
                    </body>
                </html>
                """
        await log_query(image_bytes=image_bytes, text_input=name,
                        top_match_id=top_match_id)
        return HTMLResponse(content=html)
    except Exception as e:
        await log_query(image_bytes=image_bytes, text_input=name,
                        top_match_id=None, error=str(e))
        return HTMLResponse(content=f"<h2>Error:</h2><p>{str(e)}</p>", status_code=500)


@app.get("/", response_class=HTMLResponse)
async def root_form():
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Product Matcher</title>
      <style>
          body {
            font-family: Arial, sans-serif;
            margin: 40px;
          }
          input[type="submit"] {
            padding: 8px 16px;
            font-weight: bold;
            cursor: pointer;
          }
        </style>
    </head>
    <body>
      <h2>Match Product</h2>
      <form action="/match" method="post" enctype="multipart/form-data">
        <label for="file">Upload Product Image:</label><br>
        <input type="file" name="file" required><br><br>

        <label for="name">Enter Product Description:</label><br>
        <input type="text" name="name" required><br><br>

        <input type="submit" value="Match">
      </form>
    </body>
    </html>
    """


@app.get("/logs", response_class=HTMLResponse)
async def view_logs():
    cursor = db.queries.find().sort("timestamp", -1)
    logs = await cursor.to_list(length=100)

    html_rows = ""
    for log in logs:
        timestamp = log.get("timestamp", "")
        match_id = log.get("top_match_id", "N/A")
        text = log.get("text", "")
        error = log.get("error", "")
        html_rows += f"""
        <tr>
            <td>{timestamp}</td>
            <td>{match_id}</td>
            <td>{text}</td>
            <td>{error}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Query Logs</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
            }}
            th, td {{
                border: 1px solid #ccc;
                padding: 8px 12px;
                text-align: left;
            }}
            th {{
                background-color: #f4f4f4;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
        </style>
    </head>
    <body>
        <h2>Product Matching Query Logs</h2>
        <table>
            <tr>
                <th>Timestamp</th>
                <th>Match ID</th>
                <th>Description</th>
                <th>Error</th>
            </tr>
            {html_rows}
        </table>
    </body>
    </html>
    """

    return HTMLResponse(content=html)
