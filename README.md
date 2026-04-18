# ScreenGuard Prototype

Prototipo em Python que reproduz um video e vigia a webcam ao mesmo tempo. Nos primeiros segundos mede quantos rostos estao a ver o ecra. Se depois aparecer um rosto extra durante varios frames seguidos, o video para e surge um ecra hostil.

## Como funciona

1. O video comeca a reproduzir.
2. Durante os primeiros `2.0` segundos o sistema calcula o baseline de rostos.
3. Depois disso, se a webcam detetar mais rostos do que o baseline, o sistema entra em alerta.
4. Em alerta, o video deixa de avancar e aparece um ecra hostil.
5. Carrega `R` para reiniciar o video e recalibrar. Carrega `Q` para sair.

## Instalar

Precisas de Python 3.10+ instalado localmente.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Executar

Se tiveres um video na pasta atual:

```powershell
python main.py
```

Se quiseres indicar o video manualmente:

```powershell
python main.py .\meu_video.mp4
```

Se quiseres usar uma imagem tua como ecra hostil:

```powershell
python main.py .\meu_video.mp4 --hostile-image .\hostile.png
```

## Parametros uteis

- `--camera-index 0`: escolhe a webcam.
- `--calibration-seconds 2.0`: duracao da calibracao inicial.
- `--trigger-frames 8`: quantos frames seguidos com rostos extra disparam o alerta.
- `--min-face-area-ratio 0.01`: ignora rostos muito pequenos ao fundo.
- `--edge-margin 0.08`: ignora rostos muito perto das bordas.
- `--no-fullscreen`: abre a janela em modo normal.
- `--no-loop`: nao reinicia o video quando chega ao fim.

## Observacoes

- Este prototipo nao faz reconhecimento facial. So conta quantos rostos visiveis existem em cada momento.
- Se a pessoa inicial sair e ficar outra pessoa sozinha, o sistema continua a ver `1` rosto. Ou seja, o foco aqui e detetar "mais gente a olhar" e nao "quem e a pessoa".
- Para uma instalacao real, vale a pena testar iluminacao, angulo da webcam e distancia ao ecra para ajustar os limiares.
