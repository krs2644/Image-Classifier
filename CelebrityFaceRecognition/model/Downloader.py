from icrawler.builtin import GoogleImageCrawler

celebrities = ["Emma Watson", "Robert Downey Jr", "Scarlett Johansson", "Tom Cruise", "Leonardo DiCaprio"]

for celeb in celebrities:
    folder_name = celeb.replace(" ", "_")
    print(f"Downloading images for {celeb}...")
    crawler = GoogleImageCrawler(storage={'root_dir': f'D:/projects/CelebrityFaceRecognition/model/dataset/{folder_name}'})
    crawler.crawl(keyword=celeb, max_num=50000)
