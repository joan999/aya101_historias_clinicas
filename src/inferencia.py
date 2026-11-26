"""Paso por el encoder y el decoder, y lectura de la atención cruzada."""

import torch


@torch.no_grad()
def inferir(tokenizador, modelo, prompts, historia, max_tokens, no_repeat_ngram_size=None):
    """Corre encoder y decoder sobre un lote de prompts y guarda la atención cruzada.

    no_repeat_ngram_size: si se da, el decoder no puede repetir un grupo de ese número
    de tokens. Sirve para cortar ciclos como «Sí, no. Sí, no…» (sigue siendo greedy).
    """
    extra = {"no_repeat_ngram_size": no_repeat_ngram_size} if no_repeat_ngram_size else {}
    lote = tokenizador(prompts, return_tensors="pt", padding=True, return_offsets_mapping=True)
    offsets = lote.pop("offset_mapping")
    lote = lote.to(modelo.device)

    # El encoder lee el prompt completo una sola vez
    salida_encoder = modelo.get_encoder()(input_ids=lote["input_ids"], attention_mask=lote["attention_mask"])

    # El decoder genera token por token (greedy) consultando la salida del encoder
    secuencias = modelo.generate(
        encoder_outputs=salida_encoder,
        attention_mask=lote["attention_mask"],
        max_new_tokens=max_tokens,
        do_sample=False,
        num_beams=1,
        **extra,
    )

    # Se repite el decoder con lo generado para leer la atención cruzada de cada paso
    paso = modelo(
        encoder_outputs=salida_encoder,
        attention_mask=lote["attention_mask"],
        decoder_input_ids=secuencias[:, :-1],
        output_attentions=True,
    )
    # Promedio de las cabezas en cada capa -> [lote, capas, tokens generados, tokens de entrada]
    atencion = torch.stack([capa.float().mean(dim=1) for capa in paso.cross_attentions], dim=1).cpu()

    resultados = []
    for i, prompt in enumerate(prompts):
        inicio = prompt.find(historia)
        fin = inicio + len(historia)
        # Posiciones de los tokens que pertenecen a la historia (sin instrucción ni pregunta)
        pos_historia = [j for j, (a, b) in enumerate(offsets[i].tolist()) if b > a and a >= inicio and b <= fin]
        ids_generados = secuencias[i, 1:].tolist()
        n = len(ids_generados)
        if tokenizador.eos_token_id in ids_generados:
            n = ids_generados.index(tokenizador.eos_token_id)
        ids_entrada = lote["input_ids"][i][lote["attention_mask"][i].bool()].cpu()
        # Lo que recibió el decoder para producir cada token: inicio + tokens ya generados
        ids_decoder = secuencias[i, :max(n, 1)].cpu()
        resultados.append({
            "prompt": prompt,
            "n_tokens": n,   # tokens generados antes de </s>; si llega a max_tokens, la respuesta se cortó
            "ids_entrada": ids_entrada,
            "ids_decoder": ids_decoder,
            "tokens_entrada": tokenizador.convert_ids_to_tokens(ids_entrada.tolist()),
            "tokens_decoder": tokenizador.convert_ids_to_tokens(ids_decoder.tolist()),
            "texto": tokenizador.decode(ids_generados[:n], skip_special_tokens=True).strip(),
            "tokens_salida": tokenizador.convert_ids_to_tokens(ids_generados[:n]),
            "tokens_historia": tokenizador.convert_ids_to_tokens(lote["input_ids"][i, pos_historia].tolist()),
            "atencion": atencion[i][:, :n][:, :, pos_historia],  # [capas, salida, historia]
        })
    return resultados
