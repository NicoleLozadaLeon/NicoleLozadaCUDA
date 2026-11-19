# Informe EC2: aplicaciones GPU en Equipetrol Delivery

## 3.1 Aplicación 1: tabla de distancias con Floyd–Warshall por tiles (03_memory_tiling)

### Tarea y por qué es data-parallel
- Cada ETA necesita la distancia entre dos intersecciones.
- El programa guarda una tabla all-pairs de n² entradas, en decímetros enteros.
- Referencia en CPU: Dijkstra desde cada nodo.
- En GPU: Floyd–Warshall. En la ronda k, cada par (i, j) se actualiza solo:
  `d[i][j] = min(d[i][j], d[i][k] + d[k][j])`.
- Cada ronda tiene n² actualizaciones independientes: es data-parallel.
- Las n rondas son secuenciales. La ronda k usa el resultado de la ronda k − 1.

### Dónde se usa en el programa
- El `Graph` calcula la tabla una sola vez, al arrancar.
- Con `EQ_GPU=1`, se calcula en la GPU a través del único hilo dueño de la GPU (commit de integración).

### Diseño
- **Datos.** Matriz densa de `int`, rellenada hasta un múltiplo de 32 con `kInf = 0x3fffffff`.
  Así `kInf + kInf` no desborda un `int`.
- **Movimiento.** La matriz se sube una vez. Todas las rondas corren en la GPU. Se baja una vez.
- **Block.** 32 × 32 = 1024 threads, un thread por celda. 32 es el ancho de un warp: cada fila del tile es un warp.
- **Grid por ronda k.** Tres kernels:
  - `diagonalTile`: 1 bloque, el tile (k, k).
  - `pivotRowAndColumn`: grid (bloques, 2), la fila y la columna k.
  - `remainingTiles`: grid (bloques, bloques), el resto.
- **Shared memory.** Cada bloque carga sus tiles una vez y reusa cada valor 32 veces (data reuse, sesión 13).
- **Tamaño.** Con n = 4900, N = 4928: 154 rondas × 3 = 462 launches. La fase 3 tiene 154 × 154 = 23 716 bloques por ronda.
- **CPU N threads.** `parallelFor` reparte los nodos origen en rangos contiguos. Cada thread corre Dijkstra para sus orígenes y escribe solo sus filas. No hay escrituras compartidas.

### Correctitud
- Los valores son enteros: la tabla GPU debe ser **exactamente igual** a `dijkstraOne`. Sin tolerancia.
- Los 5 tamaños dan `correct=yes` en las 450 filas del CSV.
- El grafo de prueba tiene calles de un solo sentido (una fila de cada tres). La tabla no es simétrica y se compara como grafo dirigido.
- Los pares sin camino quedan en `kInf`.
- compute-sanitizer (`results/sanitizer_laptop_fw.txt`): memcheck 0 errors, racecheck 0 hazards, synccheck 0 errors.
- En las fases 1 y 2 el tile se actualiza en el mismo lugar. Un `__syncthreads()` separa la lectura de la escritura.

### Resultados
Laptop: Intel i9-12900H (6P + 8E cores, 20 threads), RTX 3060 Laptop GPU, WSL2, Release (-O3).
Mediana de 10 repeticiones intercaladas, después de un warm-up. Tiempos en ms.
Fuente: `results/bench_laptop_fw.csv`, tabla de `analysis/report.ipynb`.

| size | CPU 1 | mejor CPU N | GPU kernel | GPU end to end | speedup kernel | speedup e2e | correct |
|---|---|---|---|---|---|---|---|
| n=100 | 0.518 | 0.376 (N=4) | 0.238 | 0.995 | 1.58 | 0.38 | yes |
| n=289 | 4.804 | 1.244 (N=14) | 0.638 | 1.798 | 1.95 | 0.69 | yes |
| n=1024 | 67.217 | 7.926 (N=20) | 4.458 | 10.617 | 1.78 | 0.75 | yes |
| n=2025 | 282.140 | 22.726 (N=20) | 26.838 | 52.956 | 0.85 | 0.43 | yes |
| n=4900 | 1791.033 | 172.834 (N=20) | 549.199 | 760.344 | 0.32 | 0.23 | yes |

![time vs size](../results/time_vs_size.png)

**Copies vs kernel.** Parte del tiempo end to end que es kernel: 24 % (n=100), 35 % (n=289), 42 % (n=1024), 51 % (n=2025), 72 % (n=4900). En tamaños chicos dominan las copias y la preparación.

![copies vs kernels](../results/copy_share.png)

**Thread sweep (n = 4900, mediana).** 1 thread: 1791 ms · 2: 927 · 4: 501 · 8: 293 · 14: 220 · 20: 173 ms.
Speedup de 10.4× con 20 threads (eficiencia 52 %). No es lineal: 8 de los 14 cores son E-cores y solo 6 tienen hyper-threading.

![thread sweep](../results/thread_sweep.png)

### Profiler (Nsight Compute, `remainingTiles`, n = 1024)
| Métrica | Valor |
|---|---|
| Theoretical occupancy | 66.67 % |
| Achieved occupancy | 63.2–63.8 % |
| Registers per thread | 25 |
| Block limit: warps / shared mem / registers / SM | 1 / 1 / 2 / 16 bloques |
| Memory throughput (unidad más cargada) | 74.6–75.6 % |
| DRAM throughput | 81–82 GB/s (24 % de 336 GB/s) |

- Un bloque de 1024 threads son 32 warps. El SM admite 48 warps: solo cabe 1 bloque. Por eso la ocupación teórica es 32/48 = 66.67 %.
- El límite son los warps y la shared memory, no los registros (25 por thread permitirían 2 bloques).
- La achieved occupancy (63–64 %) está cerca de la teórica: los warps casi no se quedan esperando.
- La DRAM solo usa el 24 % de su ancho de banda. El 75 % viene de la memoria on-chip (shared/L1). Es el efecto buscado: los tiles leen de shared, no de la global.

### Roofline
- Conteo: 2 operaciones enteras (suma y min) por celda y ronda. Tráfico global: 2 tiles de 32 × 32 por bloque y ronda.
- Intensidad aritmética: **4.0 op/byte**. Rendimiento alcanzado con n = 4900: **4.36 × 10¹¹ op/s**.
- Ridge point de la RTX 3060 Laptop: 16.1 TFLOP/s ÷ 336 GB/s ≈ 48 op/byte. Con 4 op/byte el kernel queda a la izquierda: **memory-bound**.
- El techo de memoria en 4 op/byte es 4 × 336 GB/s = 1.34 × 10¹² op/s. El kernel llega al 32 % de ese techo.
- Son operaciones enteras, no FLOPs de punto flotante.

![roofline](../results/roofline.png)

### Análisis
- **Paralelismo.** Con n = 100 la matriz es de 128 × 128: solo 16 bloques en la fase 3, menos que los 30 SMs de la GPU. La GPU no se llena. Con n = 4900 hay 23 716 bloques por ronda: sobra trabajo por launch, pero son 462 launches dependientes.
- **Overheads.** Con n = 4900, end to end − kernel = 760 − 549 = 211 ms. Incluye armar la matriz con relleno en la CPU, copiar unos 97 MB en cada sentido (4928² × 4 B) y quitar el relleno. En el programa se paga una sola vez, al arrancar.
- **Divergencia.** No hay divergencia dentro de un warp en la fase 3. Los bloques con `bi == k` o `bj == k` salen como bloque completo.
- **Contención y sincronización.** No hay atomics: cada thread es dueño de su celda. La fase 3 solo lee tiles que la fase 2 ya terminó. Por eso no necesita barreras dentro del bucle.
- **Correctitud en paralelo.** `min` entre enteros es exacto y no depende del orden. El resultado es determinista.
- **Algoritmo.** Floyd–Warshall hace N³ ≈ 1.2 × 10¹¹ actualizaciones con n = 4900. Dijkstra en una grilla de calles (m ≈ 4n aristas) hace unas n · m · log n ≈ 1.2 × 10⁹: unas 100 veces menos trabajo. Por eso la CPU con 20 threads (173 ms) le gana incluso al kernel solo (549 ms).

### Veredicto
- **Crossover end to end: nunca.** La GPU no le gana a la mejor CPU en ningún tamaño medido.
- El kernel solo gana hasta n = 1024 (1.6–1.9×). Desde n = 2025 pierde también.
- **Decisión: mantener la CPU** (Dijkstra multithread) para esta tabla. Se calcula una vez al arrancar y tarda milisegundos, tanto a la escala del brief (32 nodos) como en la ciudad (1600 nodos).

### Threats to validity
- **Dispersión de la GPU.** Con n = 4900, el end to end fue de 642 a 801 ms y el kernel de 404 a 562 ms.
- **Un valor atípico.** Con n = 2025 hubo una repetición de 233 ms end to end, contra una mediana de 53 ms. Por eso se reporta la mediana y no el promedio.
- La CPU va estable: con n = 4900 y 20 threads, de 167 a 179 ms.
- La CPU y la GPU de la laptop comparten potencia y temperatura.
- En WSL2 cada launch y cada copia pasa por la capa WDDM de Windows.

### Cómo reproducir
```bash
./build-gpu/delivery_sim config/equipetrol.json --bench results/bench_laptop_fw.csv --machine laptop --threads 1,2,4,8,14,20 --apps fw
for tool in memcheck racecheck synccheck; do
  compute-sanitizer --tool $tool ./build-gpu/delivery_sim config/equipetrol.json --bench /tmp/s.csv --size 0 --reps 1 --threads 2 --apps fw
done
ncu --set full -k regex:"remainingTiles" -c 2 --csv --page details ./build-gpu/delivery_sim config/equipetrol.json --bench /tmp/n.csv --size 2 --reps 1 --threads 2 --apps fw
cd analysis && jupyter nbconvert --to notebook --execute --allow-errors --inplace report.ipynb
```

## 3.2 Aplicación 2: ETA en lote con los repartidores como SoA (04_performance)

### Tarea y por qué es data-parallel
- El Dispatcher necesita el ETA de cada repartidor para cada pedido.
- ETA = el repartidor queda libre, va al restaurante, espera la comida, entrega. Es la misma fórmula del Dispatcher, en enteros.
- Con m pedidos y k repartidores hay m × k pares. Cada par se calcula solo, sin depender de otro: es data-parallel.
- Referencia en CPU: un doble bucle pedido × repartidor.

### Dónde se usa en el programa
- El Dispatcher, al elegir repartidor para un pedido.
- Con `EQ_GPU=1`, la GPU calcula el ETA de todos los repartidores disponibles y la aplicación 3 elige el mejor sin volver a la CPU (commit de integración).

### Diseño
- **Datos SoA.** Los repartidores van como dos arreglos: `freeNode[k]` (4 B) y `freeAtMs[k]` (8 B).
- **Datos AoS (para comparar).** Una ficha `CourierAoS` de 32 bytes por repartidor, con 7 campos. El kernel usa solo 2: 12 de los 32 bytes.
- **Tabla de distancias.** `DeviceTable` la sube una vez y la deja en la GPU. En el bench no se cuenta en el end to end, porque así funciona en el programa.
- **Block.** 256 threads (8 warps), en x sobre los repartidores. 256 es múltiplo de 32 y deja 6 bloques por SM (48 warps): ocupación teórica 100 %.
- **Grid.** (⌈k / 256⌉, m): `x` recorre repartidores, `y` recorre pedidos. Un thread por par.
- **Coalescing.** Los 32 threads de un warp leen 32 repartidores **consecutivos**:
  - SoA: un warp lee 128 B de `freeNode` y 256 B de `freeAtMs`, todo útil.
  - AoS: un warp toca 32 fichas × 32 B = 1 KB para usar 384 B.
- **Lo que no es coalesced en ninguna versión.** `dist[freeNode * n + restaurante]` salta a filas distintas de la tabla (gather). Es igual en SoA y en AoS.
- **Escritura.** `out[i * k + j]`: consecutiva en el warp, coalesced en las dos versiones.
- **CPU N threads.** `parallelFor` reparte los pedidos en rangos. Cada thread escribe solo sus filas.

### Correctitud
- Los ETA son enteros: GPU (SoA y AoS) debe ser **exactamente igual** a `cpuOne`. Sin tolerancia.
- Los 5 tamaños dan `correct=yes` en todas las filas del CSV.
- compute-sanitizer (`results/sanitizer_laptop_eta.txt`): memcheck 0 errors, racecheck 0 hazards, synccheck 0 errors.
- No hay shared memory ni barreras: cada thread lee y escribe solo lo suyo.

### Resultados
Laptop: Intel i9-12900H (6P + 8E cores, 20 threads), RTX 3060 Laptop GPU, WSL2, Release (-O3).
Mediana de 10 repeticiones intercaladas, después de un warm-up. Tiempos en ms. GPU = versión SoA.
Fuente: `results/bench_laptop_eta.csv`, tabla de `analysis/report.ipynb`.

| size (pedidos × repartidores) | CPU 1 | mejor CPU | GPU kernel | GPU end to end | speedup kernel | speedup e2e | correct |
|---|---|---|---|---|---|---|---|
| 10×50 | 0.005 | 0.005 (CPU 1) | 0.011 | 0.577 | 0.42 | 0.01 | yes |
| 100×500 | 0.350 | 0.275 (N=4) | 0.020 | 0.698 | 13.8 | 0.39 | yes |
| 500×2000 | 8.255 | 2.172 (N=14) | 0.080 | 2.209 | 27.2 | 0.98 | yes |
| 1000×5000 | 38.822 | 6.690 (N=20) | 0.341 | 5.321 | 19.6 | 1.26 | yes |
| 2000×5000 | 90.061 | 30.497 (N=20) | 0.665 | 25.748 | 45.9 | 1.18 | yes |

- **Tamaño del problema** = pares pedido × repartidor: de 500 a 10⁷, más de cuatro órdenes de magnitud.
- **Misma máquina.** CPU y GPU son de la misma laptop: esta tabla ya es la comparación «same-machine».

![time vs size](../results/time_vs_size.png)

**SoA vs AoS (kernel, mediana).**

| size | SoA kernel (ms) | AoS kernel (ms) | AoS / SoA | ancho de banda efectivo SoA | AoS |
|---|---|---|---|---|---|
| 500×2000 | 0.080 | 0.086 | 1.08 | 250 GB/s | 233 GB/s |
| 1000×5000 | 0.341 | 0.467 | 1.37 | 293 GB/s | 214 GB/s |
| 2000×5000 | 0.665 | 0.878 | 1.32 | 301 GB/s | 228 GB/s |

- Ancho de banda efectivo = bytes útiles por par (20 B: `freeNode` 4 + `freeAtMs` 8 + una entrada de la tabla 4 + la salida 4) ÷ tiempo del kernel.
- Desde 1000×5000, **SoA es 32–37 % más rápido** que AoS.
- En tamaños chicos (10×50, 100×500) los kernels duran 10–20 µs: la diferencia es ruido de medición.

**Copies vs kernel (SoA).** Parte del end to end que es kernel: 2.0 % (10×50), 2.9 %, 3.6 %, 6.4 % (1000×5000), 2.6 % (2000×5000). **Más del 93 % es copia y reserva.**

![copies vs kernels](../results/copy_share.png)

**Thread sweep (2000×5000, mediana).** 1 thread: 90.1 ms · 2: 53.8 · 4: 36.5 · 8: 32.5 · 14: 30.6 · 20: 30.5 ms.
Speedup de solo 3.0× con 20 threads. Desde 8 threads casi no mejora: la CPU también es memory-bound (escribe 40 MB de salida y lee la tabla al azar).

![thread sweep](../results/thread_sweep.png)

### Profiler (Nsight Compute, 500×2000, grid (8, 500), block 256)
| Métrica | `etaSoA` | `etaAoS` |
|---|---|---|
| Duration | 76.7 µs | 80.4 µs |
| Theoretical / achieved occupancy | 100 % / 86.5 % | 100 % / 87.3 % |
| Block limit: warps / registers / shared mem | 6 / 8 / 8 bloques | 6 / 8 / 8 bloques |
| Registers per thread | 30 | 30 |
| Waves per SM | 22.2 | 22.2 |
| Compute (SM) throughput | 42.7 % | 40.5 % |
| DRAM throughput | 51.3 % (172 GB/s) | 49.3 % (165 GB/s) |
| L2 throughput · L2 hit rate | 50.1 % · 78.3 % | 70.0 % · 85.4 % |
| L1/TEX throughput · L1 hit rate | 64.7 % · 22.9 % | 67.2 % · 43.4 % |
| Avg. active threads per warp | 23.8 de 32 | 23.7 de 32 |
| Branch efficiency · divergent branches | 88.8 % · 139 | 88.8 % · 139 |

- **Occupancy.** Bloques de 256 threads = 8 warps. Caben 6 bloques por SM (48 warps): teórica 100 %. Limitan los warps. La alcanzada (86–87 %) está cerca.
- **AoS mueve más datos.** La L2 trabaja al 70 % en AoS contra 50 % en SoA, para el mismo trabajo. Son las fichas de 32 B que se traen enteras.
- **La diferencia es moderada** (5 % en este tamaño, 32–37 % en los grandes). El costo grande es el mismo en las dos versiones: el gather a la tabla de distancias y la escritura de la salida.
- **Divergencia.** Branch efficiency 88.8 % y unos 24 de 32 threads activos por warp. Lo más probable es que venga de las dos divisiones de 64 bits (`* 360000 / metersPerHour`): la GPU no tiene una instrucción de división de 64 bits, y el compilador genera una rutina con saltos que dependen de los valores. Además, 2000 repartidores en bloques de 256 dejan 48 threads sin trabajo por fila (2.3 %).

### Roofline
- Conteo por par: unas 12 operaciones (sumas, max, multiplicaciones y divisiones) y 20 bytes útiles.
- Intensidad aritmética: **0.6 op/byte**. Rendimiento alcanzado en 2000×5000: **1.8 × 10¹¹ op/s**.
- Ridge point de la RTX 3060 Laptop ≈ 48 op/byte. Con 0.6 el kernel queda muy a la izquierda: **memory-bound**.
- El techo de memoria en 0.6 op/byte es 0.6 × 336 GB/s = 2.0 × 10¹¹ op/s. El kernel llega al **90 %** de ese techo.

![roofline](../results/roofline.png)

### Análisis
- **Paralelismo.** La GPU tiene 30 SMs × 1536 threads = 46 080 threads residentes. Con 10×50 hay 10 bloques (2560 threads): no llena la GPU. Desde 500×2000 (10⁶ pares) hay 22 waves: la llena de sobra.
- **Overheads.** El kernel es menos del 7 % del end to end. Lo demás es subir los repartidores, reservar la salida y copiar de vuelta la matriz de ETA: 2000 × 5000 × 4 B = 40 MB. De 1000×5000 a 2000×5000 la salida se duplica, pero el end to end crece casi 5× (5.3 → 25.7 ms): la reserva y la copia de 40 MB no escalan lineal.
- **¿Los datos pueden quedarse en la GPU?** Sí, y es lo que hace el programa. La tabla de distancias se sube una vez (`DeviceTable`). La matriz de ETA nunca vuelve: la aplicación 3 la reduce en la GPU y devuelve 8 bytes (el mejor repartidor).
- **Memory-bound.** Pocas operaciones por byte, como muestra el roofline. Por eso importa cómo se leen los datos (SoA contra AoS).
- **Optimización antes/después.** La comparación SoA (después) contra AoS (antes) es la optimización del capítulo: 32–37 % menos tiempo de kernel en los tamaños grandes.
- **Contención y sincronización.** No hay atomics ni barreras: cada thread escribe su propia celda.
- **Correctitud en paralelo.** Aritmética entera: el resultado no depende del orden. Determinista.

### Veredicto
- **Crossover end to end: 1000×5000 (5 × 10⁶ pares).** Desde ahí la GPU le gana a la mejor CPU: 1.26× y 1.18×. En 500×2000 es un empate (0.98×).
- El kernel solo gana desde 100×500 (14–46×). Lo que frena el end to end son las copias.
- **Escala del brief (8 repartidores, un pedido por ranking):** 8 pares. La CPU tarda microsegundos y la GPU paga 0.5 ms solo en copias y lanzamiento. **CPU.**
- **Ciudad (un pedido contra 400 o 2000 repartidores):** de 400 a 2000 pares por ranking, muy por debajo del crossover. **CPU**, salvo que se calculen muchos pedidos juntos.
- **Decisión: GPU solo por encima de un umbral** (unos 5 millones de pares por lote), y sin devolver la matriz: encadenada con la aplicación 3.

### Threats to validity
- **Kernels muy cortos.** En 10×50 y 100×500 el kernel dura 10–20 µs. Ahí la medición tiene ruido (AoS parece más rápido); por eso se compara desde 500×2000.
- **Dispersión.** En 2000×5000: GPU end to end de 25.3 a 28.4 ms; CPU con 20 threads de 27.1 a 32.0 ms. Los rangos casi se tocan: la ventaja de 1.18× es chica.
- **Divergencia sin confirmar en el código fuente.** La causa (división de 64 bits) es la explicación más probable; se puede confirmar en la vista Source de Nsight Compute.
- La CPU y la GPU de la laptop comparten potencia y temperatura. En WSL2 cada launch y cada copia pasa por la capa WDDM de Windows.

### Cómo reproducir
```bash
./build-gpu/delivery_sim config/equipetrol.json --bench results/bench_laptop_eta.csv --machine laptop --threads 1,2,4,8,14,20 --apps eta
for tool in memcheck racecheck synccheck; do
  compute-sanitizer --tool $tool ./build-gpu/delivery_sim config/equipetrol.json --bench /tmp/s.csv --size 0 --reps 1 --threads 2 --apps eta
done
ncu --set full -k regex:"etaSoA|etaAoS" -c 2 --csv --page details ./build-gpu/delivery_sim config/equipetrol.json --bench /tmp/n.csv --size 2 --reps 1 --threads 2 --apps eta
cd analysis && jupyter nbconvert --to notebook --execute --allow-errors --inplace report.ipynb
```
