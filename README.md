# ScreenGuard Prototype

Prototipo em Python que reproduz um video com imagem e som enquanto vigia a webcam ao mesmo tempo. Em vez de apenas contar quantos rostos estao a olhar para o ecra, o sistema aprende quem esta autorizado durante a calibracao inicial e tenta reconhecer a pessoa com embeddings faciais reais. Se aparecer outra pessoa, o video para, o audio pausa e surge um ecra hostil.

## Como funciona

1. O video comeca a reproduzir com a respetiva faixa de audio.
2. Durante os primeiros `2.0` segundos o sistema aprende perfis de identidade dos rostos visiveis.
3. Depois disso, cada rosto detetado e comparado com os perfis autorizados.
4. Se aparecer uma pessoa nao autorizada ou desconhecida durante varios frames seguidos, o sistema entra em alerta.
5. Em alerta, o video deixa de avancar, o audio pausa e aparece um ecra hostil onde o intruso pode desenhar movendo o nariz, agora com estabilizacao para reduzir tremores.
6. Carrega `S` para gravar o desenho, `C` para limpar, `R` para reiniciar o video, voltar o audio ao inicio e recalibrar, ou `Q` para sair.

## Instalar

Precisas de Python 3.10+ instalado localmente.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Na primeira execucao, o programa extrai temporariamente a faixa de audio do video para a reproduzir em paralelo com os frames mostrados no OpenCV.

Tambem na primeira execucao, se os modelos ONNX do YuNet e SFace nao estiverem na pasta `models/`, o programa tenta descarrega-los automaticamente para essa pasta.

## Estrutura do codigo

- `main.py`: ponto de entrada. Serve apenas para iniciar o programa.
- `screenguard_app.py`: compatibilidade com a versao anterior, encaminha para o pacote novo.
- `screenguard/app.py`: ciclo principal do programa.
- `screenguard/config.py`: argumentos e escolha do video.
- `screenguard/media.py`: janela, webcam, video, imagens e audio.
- `screenguard/guard.py`: regras de calibracao, vigilancia e disparo do alerta.
- `screenguard/face.py`: modelos, detecao facial e reconhecimento.
- `screenguard/drawing.py`: desenho controlado pelo nariz.
- `screenguard/ui.py`: textos, paineis e ecra de alerta.
- `screenguard/models.py`: classes de dados usadas pelo programa.
- `screenguard/references.py`: imagens externas e fotos de referencia.
- `screenguard/utils.py`: funcoes pequenas usadas por varios ficheiros.
- `models/`: pasta dos modelos ONNX usados pelo OpenCV.

## Referencias opcionais por pessoa

Se quiseres que o sistema mostre nomes reais em vez de rotulos temporarios como `Pessoa 1`, podes fornecer uma pasta com imagens de referencia, organizada assim:

```text
referencias/
  Ana/
    foto1.jpg
    foto2.jpg
  Bruno/
    frente.png
```

Cada subpasta corresponde a uma pessoa. O sistema tenta extrair embeddings faciais a partir dessas fotos e usar o nome da pasta como etiqueta.

## Executar

Se tiveres um video na pasta atual:

```powershell
python main.py
```

Se quiseres indicar o video manualmente:

```powershell
python main.py .\meu_video.mp4
```

Se quiseres usar referencias nomeadas:

```powershell
python main.py .\meu_video.mp4 --known-faces-dir .\referencias
```

Se quiseres guardar os modelos numa pasta diferente:

```powershell
python main.py .\meu_video.mp4 --models-dir .\meus_modelos
```

Se quiseres usar uma imagem tua como ecra hostil:

```powershell
python main.py .\meu_video.mp4 --hostile-image .\hostile.png
```

## Parametros uteis

- `--camera-index 0`: escolhe a webcam.
- `--calibration-seconds 2.0`: duracao da calibracao inicial.
- `--trigger-frames 8`: quantos frames seguidos com uma pessoa nao autorizada disparam o alerta.
- `--recognition-threshold 0.363`: limiar cosine do SFace para considerar que dois rostos sao da mesma pessoa.
- `--min-profile-observations 3`: numero minimo de observacoes para um perfil ficar autorizado.
- `--known-faces-dir .\referencias`: pasta com fotos de referencia por pessoa.
- `--models-dir .\models`: pasta onde os modelos ONNX sao guardados.
- `--drawings-dir .\intruder_drawings`: pasta onde os desenhos do intruso sao gravados ao carregar `S`.
- `--min-face-area-ratio 0.01`: ignora rostos muito pequenos ao fundo.
- `--edge-margin 0.08`: ignora rostos muito perto das bordas.
- `--no-fullscreen`: abre a janela em modo normal.
- `--no-loop`: nao reinicia o video quando chega ao fim.

## Observacoes

- Sem `--known-faces-dir`, o sistema continua a reconhecer identidades aprendidas na sessao atual, mas usa nomes temporarios como `Pessoa 1`, `Pessoa 2`, etc.
- As identidades aprendidas durante a calibracao sao consideradas autorizadas ate carregares `R` ou fechares o programa.
- O reconhecimento agora usa YuNet para detecao e SFace para embeddings faciais, o que e bastante mais robusto do que a assinatura visual anterior.
- Se o download automatico falhar, podes colocar manualmente `face_detection_yunet_2023mar.onnx` e `face_recognition_sface_2021dec.onnx` dentro da pasta `models/` ou apontar outra pasta com `--models-dir`.
- Se o sistema estiver demasiado sensivel ou demasiado permissivo, ajusta `--recognition-threshold` e `--min-profile-observations`.
- Para uma instalacao real, vale a pena testar iluminacao, angulo da webcam e distancia ao ecra para ajustar os limiares.
- Se o ficheiro de video nao tiver uma faixa de audio valida, o prototipo continua a funcionar, mas fica sem som.
