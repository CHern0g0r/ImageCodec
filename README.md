# Сжатие изображения при помощи автокодировщика
## Автор: Черногорский Федор Евгеньевич

## Модель

В качестве модификации моделей энкодера и декодера были удалены слои пулинга, а также были использованы Residual блоки.


## Запуск

Для запуска процесса обучения нужно запустить скрипт `train.sh`.

Для запуска тестов нужно выполнить `python -m src.CNNImageCodec`.


## Результаты

**MEAN SSIM: 0.53**

**MEAN BPP: 0.34 vs 0.49**

![](doc/result.png)

![](doc/q_bpp.png)

![](doc/perimage.png)


*Предложенный плюсовый код не компилится на arch без флага '-lsupc++'*