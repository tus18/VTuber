from icrawler.builtin import BingImageCrawler
#情報を入力
select_word = input("ほしい画像を教えてください:")
select_num = input("何枚ほしいですか？:")
#ダウンロード先のフォルダーを指定する
crawler = BingImageCrawler(storage={"root_dir": "./img"})
#google検索＆ダウンロード
#検索キーワードとダウンロード数を決定する
crawler.crawl(keyword=select_word, max_num=int(select_num))