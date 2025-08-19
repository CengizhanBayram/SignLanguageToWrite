# SignLanguageToWrite
## Project Description
This project includes an application that aims to detect the sign language of sign language users and translate this sign language into text. It was developed using libraries such as OpenCV and TensorFlow.
Her zaman içimde kalan projeme dönüp bakmak istedim 
# Requirements
The following requirements are needed to run the project:

Python 3.11: The project is compatible with Python 3.11 versions.
OpenCV: OpenCV library must be installed for image processing. 
TensorFlow: TensorFlow is required to run the deep learning model. 
Switch on your camera:
The application will use your camera to recognise sign language. The camera window will open automatically.

# Observe Sign Language Translation:
Follow the sign language in the image taken by the camera and observe the translation results of the application.
This christimas is my begining of this project
![image](https://github.com/CengizhanBayram/SignLanguageToWrite/assets/110194474/2438c6b8-b19f-493d-8ee6-33b25680d0c6)
Proje Mimarisi: İşaret Dili Tanıma İçin Neden RNN ve LSTM Kullandık?
Bu projede, işaret dili hareketlerini metne çevirmek için Tekrarlayan Sinir Ağı (RNN) konseptini temel alan ve bu konseptin en gelişmiş uygulamalarından biri olan Uzun Kısa Süreli Bellek (LSTM) mimarisini kullandık. Bu seçimin arkasındaki temel neden, işaret dilinin doğasıyla doğrudan ilgilidir: İşaret dili, statik fotoğraflar bütünü değil, zamana bağlı bir hareketler dizisidir.

1. Temel Sorun: İşaret Dili Bir "Zaman Serisi" Verisidir
Geleneksel bir sinir ağı (CNN gibi), tek bir görüntüyü analiz etmede çok başarılıdır. Örneğin, bir elin "A" harfi şeklinde durduğu bir fotoğrafı kolayca sınıflandırabilir. Ancak işaret dili bundan çok daha fazlasıdır:

Hareketin Yönü ve Hızı: "Teşekkürler" işareti, elin çeneden başlayıp ileri doğru hareket etmesiyle anlam kazanır. Sadece elin başlangıç veya bitiş pozisyonu yeterli değildir.

Sıralama (Grammar of Motion): Tıpkı kelimelerin bir araya gelerek cümle oluşturması gibi, işaretler de belirli bir sırada ve akışta birleşerek anlamlı ifadeler oluşturur. Bir hareketin önceki ve sonraki adımları, o anki anlamı doğrudan etkiler.

Bu nedenle, projemizin temelindeki sorun, tek bir kareyi değil, kareler arasındaki sıralı ilişkiyi ve zamansal bağlamı anlayabilen bir modele ihtiyaç duymamızdı. İşte bu noktada RNN'ler devreye girer.

2. Çözümün İlk Adımı: RNN ve "Hafıza" Konsepti
Tekrarlayan Sinir Ağları (RNNs), "hafızaya" sahip olacak şekilde tasarlanmıştır. Bir dizideki her bir elemanı (bizim durumumuzda her bir video karesini) işlerken, bir önceki elemandan öğrendiği bilgileri de bir sonraki adıma taşır.

Nasıl Çalışır?: Model, 30 karelik bir dizinin 5. karesindeki eklem noktalarına bakarken, ilk 4 karede elin ve vücudun nerede olduğuna dair bir "özet bilgiye" sahiptir. Bu sayede, hareketin nereden gelip nereye gittiğini anlar ve basit bir akışı takip edebilir.

Ancak standart RNN'lerin önemli bir zayıflığı vardır: Kısa süreli hafıza.

3. Hafızayı Güçlendirmek: LSTM ve "Akıllı Kapılar"
Standart RNN'ler, dizi uzadıkça en baştaki önemli bilgileri unutma eğilimindedir. Buna teknik olarak "Ufuklayan Gradyan Problemi" (Vanishing Gradient Problem) denir. Örneğin, 10 saniyelik bir işaret dili cümlesinde, model cümlenin başındaki yüz ifadesinin sondaki anlamı nasıl etkilediğini unutabilir.

Uzun Kısa Süreli Bellek (LSTM), bu sorunu çözmek için tasarlanmış özel bir RNN türüdür. LSTM'in sırrı, "kapı mekanizmaları" (gate mechanisms) adı verilen akıllı iç yapısıdır. Bu kapılar, bilginin hücre içinde nasıl akacağını kontrol eder:

Unutma Kapısı (Forget Gate): Hücredeki eski bilgilerden hangilerinin artık önemsiz olduğuna ve atılması gerektiğine karar verir. (Örn: "Bir önceki işaret bitti, o işarete ait el pozisyonunu artık unutabilirim.")

Giriş Kapısı (Input Gate): Mevcut kareden gelen yeni bilgilerden hangilerinin önemli olduğuna ve hafızaya eklenmesi gerektiğine karar verir. (Örn: "El şimdi yüz hizasına geldi, bu bilgi önemli, bunu kaydetmeliyim.")

Çıkış Kapısı (Output Gate): Hücrede depolanan bilgilerden hangisinin, o anki tahmin için kullanılacağına karar verir. (Örn: "Elim çenemde ve ileri doğru hareket ediyor, hafızamdaki bu bilgi 'teşekkürler' anlamına geliyor, çıktım bu olmalı.")

Bu akıllı kapılar sayesinde LSTM, bir hareket dizisindeki hem kısa vadeli (anlık parmak bükülmesi gibi) hem de uzun vadeli (cümlenin başındaki bir jestin sonunu etkilemesi gibi) bağımlılıkları başarıyla öğrenir.

Sonuç: Projemiz İçin Neden Mükemmel Bir Seçim?
Özetle, RNN ve LSTM kombinasyonunu seçtik çünkü:

RNN Konsepti, işaret dilinin zamana bağlı doğasını modellemek için gerekli olan temel "hafıza" fikrini sunar.

LSTM Mimarisi, bu hafızayı çok daha güçlü ve seçici hale getirerek, kısa ve uzun süreli hareket desenlerini ayırt etmemizi sağlar.

Bu mimari, modelimizin sadece statik pozları "ezberlemesini" değil, aynı zamanda işaret dilinin akıcı "gramerini" ve hareketler arasındaki karmaşık ilişkileri "anlamasını" mümkün kılar. Bu da projemizin doğruluğu ve güvenilirliği için kritik bir öneme sahiptir.
