# Equipetrol Delivery

Map data © OpenStreetMap contributors (ODbL).

This program simulates couriers, restaurants, a dispatcher and orders in Zona Equipetrol, in accelerated time and drawn live over the real OpenStreetMap image of the zone. Every courier, the dispatcher and the order generator run at the same time on their own threads, and the estimates for each order are computed in parallel by a thread pool. The goal is to serve the orders without losing, duplicating or corrupting any, which is checked with ThreadSanitizer and a log checker.

![The simulation window](docs/window.png)



## 1. Build and run

Dependencies (the versions I used):

| Tool / library | Version | Used for |
|---|---|---|
| Ubuntu (WSL2, kernel 6.6.87.2) | 26.04.1 LTS | build and run environment |
| g++ (includes ThreadSanitizer, AddressSanitizer, LeakSanitizer, UBSan) | 15.2.0 | compiler and the tools of section 6 |
| CMake | 4.2.3 | build |
| GNU make | 4.4.1 | build |
| OpenCV | 4.10.0 | the window only |
| Qt5 (OpenCV's window backend on this system) | 5.15.18 | pulled in by OpenCV, not used directly |
| nlohmann/json (vendored in `third_party/`) | 3.11.3 | reading the config, writing the log |
| Python | 3.14.4 | tools only (log checker, experiments) |

Build and run exactly as the brief says:

```sh
mkdir build && cd build
cmake .. && make
./delivery_sim ../config/equipetrol.json
./delivery_sim ../config/equipetrol.json --log run1.log
```

ThreadSanitizer build (same target, instrumented):

```sh
mkdir build-tsan && cd build-tsan
cmake -DENABLE_TSAN=ON .. && make
```

`-DENABLE_ASAN=ON` gives an AddressSanitizer + LeakSanitizer build the same way.

Stop the simulation with Ctrl+C (SIGINT), `kill` (SIGTERM), or by closing the window. The last line of stdout is the JSON summary.

### Experiment settings

The command line is exactly the one of the brief. The experiments change the design through environment variables, read once at start-up (`src/core/Settings.h`). With none set, the program runs the final design.

| Variable | Values | Default |
|---|---|---|
| `EQ_ESTIMATES` | `pool`, `async`, `deferred` | `pool` |
| `EQ_CANDIDATES` | couriers compared per order (K) | `8` |
| `EQ_FOOD_WAIT` | `before` (wait for food, then take a slot), `holding` | `before` |
| `EQ_SLOW_ESTIMATE_MS` | first estimate of each order sleeps this long (fault injection) | `0` |
| `EQ_FAIL_ESTIMATE` | probability that one estimate throws (fault injection) | `0` |
| `EQ_GRAPH_TASKS` | tasks for the shortest-path tables (0 = cores) | `0` |
| `EQ_GRAPH_GRAIN` | sources claimed at a time (0 = static chunks) | `4` |

### Tests

```sh
python3 tools/make_test_configs.py                     
tools/run_scenario.sh build-tsan/delivery_sim config/scenarios/stress.json stress 0
tools/run_scenario.sh build/delivery_sim config/equipetrol.json real 8 INT
python3 tools/check_log.py runs/real.log runs/real.summary config/equipetrol.json
```

`tools/check_log.py` checks a log and its summary against the rules of sections 4, 6 and 7: one final state per order, no order in two bags, bag capacity, pickup slots, time never decreasing for an order, valid JSON lines, and summary equal to the log.

## 2. Project layout

The sources are grouped by their role in the concurrency design. Each folder only uses the folders below it: `view`, `app` → `actors` → `parallel`, `shared` → `core`.

| Folder | What lives there | Files |
|---|---|---|
| `src/` | entry point: signals, start line, frame loop, summary | `main.cpp` |
| `src/core/` | data that is read-only during the run, or a plain value: no lock needed | `Config`, `Settings.h`, `Graph`, `SimClock.h`, `Order.h` |
| `src/shared/` | monitors: shared data together with its mutex (and condition variable) | `EventLog`, `OrderBook`, `Restaurant` |
| `src/actors/` | objects whose code runs on their own threads | `Courier`, `Fleet`, `Dispatcher`, `OrderGenerator` |
| `src/parallel/` | the thread pool for the estimates | `EstimatePool.h` |
| `src/app/` | owns every object and thread; start and stop | `Simulation`, `SanitizerSuppressions.cpp` |
| `src/view/` | the OpenCV window, drawn by the main thread from snapshots | `Window` |

## 3. Program schematic

Every thread of execution, what it loops over, and every channel through which two threads communicate or wait for each other.

```mermaid
flowchart TB
  M["main thread<br/>loop: draw a frame (33 ms), check the stop flag and the duration"]
  G["generator thread<br/>loop: wait_until next arrival, create 1 order or a burst"]
  D["dispatcher thread<br/>loop: take an order, run K estimates, offer it"]
  P["estimate pool: max(cores, K) worker threads<br/>loop: wait for a job, run it outside the lock"]
  C["courier threads, one per courier<br/>loop: wait for work, ride, take a slot, collect, deliver"]
  B[("OrderBook<br/>mutex + condition variable")]
  Q[("pool job queue<br/>mutex + condition variable")]
  R[("Restaurant x6<br/>mutex + condition variable, tickets")]
  F[("Fleet<br/>mutex + condition variable, version")]
  L[("EventLog<br/>mutex, innermost lock")]
  S{{"g_stopRequested<br/>lock-free atomic, set by SIGINT/SIGTERM"}}

  G -->|"submit(order), by move"| B
  B -->|"take() blocks on the condition variable"| D
  D -->|"submit(estimate)"| Q
  Q --> P
  P -.->|"future: value or exception, wait_until(deadline)"| D
  D -->|"offer(order) under the courier lock"| C
  D -->|"waitForChange(seen, deadline)"| F
  C -->|"notifyChanged(), handOver()"| F
  C -->|"acquireSlot(), released by SlotGuard"| R
  C -->|"transferAll: scoped_lock on both couriers"| C
  G -->|"orderCreated, orderRejected"| L
  D -->|"orderRejected noCourier"| L
  C -->|"assigned, pickup, delivered, breakdown, reassigned"| L
  S -.->|"read once per frame"| M
  M -->|"start: one promise per thread; stop: close, notify_all, join"| G
  M -->|"start / stop"| D
  M -->|"start / stop"| C
  M -->|"shutdown: notify_all, join"| P
  M -.->|"snapshots: view(), deliveryNodes(), summary()"| C
```

**When a thread has nothing to do, it sleeps.** No thread ever loops asking "is there work yet?". Each one waits on something that wakes it up:

- A **courier** with an empty bag sleeps on its own condition variable (`Courier::nextTask()`). The dispatcher wakes it when it gives it an order.
- The **dispatcher** sleeps in `OrderBook::take()` until an order arrives. If no courier can take an order yet, it sleeps in `Fleet::waitForChange()` until a courier gets room in its bag or is repaired.
- The **order generator** sleeps until the moment of the next order (`wait_until`).
- The **pool workers** sleep until the dispatcher puts an estimate in their queue.
- The **main thread** draws one frame of the window every 33 ms.

**How the program stops.** On Ctrl+C, SIGTERM, the end of the run or closing the window, `Simulation::stop()` stops everything in this order:

1. It stops the order generator first, so no new orders appear while the rest is shutting down.
2. It wakes every sleeping thread: it closes the order book, stops the fleet and every courier, and stops the restaurants.
3. It waits for every thread to finish (`join`): the dispatcher, all couriers and the pool workers.
4. It writes `simulationStopping` as the last line of the log. Orders that were not finished are counted as `pending` in the summary.

## 4. Concept map

For every row: the file and the place in the code, and in one or two sentences why the concept is the right one there and what would go wrong without it.

| Folder | Concept that must be present | Where (file, place) | Why it is right here, and what goes wrong without it |
|---|---|---|---|
| 01 | Explicit thread creation and a deliberate fate for every thread | `Simulation::start()` creates the threads, `Simulation::stop()` joins them, `~Simulation()` calls `stop()`; `EstimatePool` constructor and `shutdown()` | Every courier must work on its own and at the same time as the others, so each one gets its own thread. Every thread is joined before the program ends; without that, a thread could be killed in the middle of its work, or the program would crash with `std::terminate`. |
| 01 | A decision that depends on the hardware's available parallelism | `Graph::Graph` (number of tasks), `poolSize()` in `Simulation.cpp` | More busy threads than cores do not make the work faster, so the number of tasks and pool workers comes from `hardware_concurrency()` (2 if it returns 0).The pool gets the larger of the cores and K, so that all K estimates of one order can run at the same time. A fixed number would waste cores on a big machine and overload a small one. |
| 02 | Data handed to a thread by value, by reference, and by move | by value: the estimate lambda in `Dispatcher::rankCouriers()`; by reference: `std::ref(dispatcher_)` in `Simulation::start()`, `std::ref(nextSource)` in `Graph::Graph`; by move: the promises and the orders | By value: each estimate works on its own copy of the courier, so it cannot see the courier change while it computes. By reference: the dispatcher and the counter are shared on purpose and live longer than the threads that use them. By move: a promise or an order must exist in only one place. |
| 02 | Orders by move semantics, in one place at a time | `OrderBook::submit/take`, `Courier::offer`, `Courier::transferAll`, `Order.h` | An order moves from the generator to the order book, to the dispatcher, to a courier's bag, and after a breakdown to another bag. It is never copied, so it can never be delivered twice or lost. |
| 03 | Member functions of shared objects on threads; containers of threads | `&Courier::run` with `courierThreads_` in `Simulation::start()`; `EstimatePool::workers_` | Each courier thread runs that courier's own `run()` method. All the threads are kept in one vector so they can be joined in a loop; threads cannot be copied, so they are added with `emplace_back`. |
| 03 | Shared ownership that guarantees lifetime; no owning raw pointers | `std::shared_ptr<Courier>` in `Fleet`, copied into each courier thread; `std::shared_ptr<Restaurant>` | Each courier thread holds a copy of its courier's `shared_ptr`, so the courier object stays alive as long as its thread runs. With a raw pointer, we would have to prove by hand that the object outlives the thread. |
| 04 | Tasks that return a result through a future, with a stated launch policy | `std::async(std::launch::async, ...)` in `Graph::Graph`; `EstimatePool::submit()` (`packaged_task`) | The shortest-path tables are built by tasks started with `std::launch::async`, which forces real new threads; the default policy could run them one after another. Each estimate returns its result, or its error, through a future. |
| 04 | A promise/future pair as a one-shot channel | Simulation::start(): one promise<void> for each courier thread, the dispatcher and the generator, before the start line  | Each thread fulfils its promise when it is ready, and the main thread waits for all of them before printing "Simulation has started...". Waiting with a sleep would be a guess: on a slow machine the line could appear too early. |
| 04 | A wait with a time limit, and an exception that crosses threads | `Dispatcher::rankCouriers()`: `wait_until(deadline)` and `get()` inside try/catch | The dispatcher waits for the estimates no longer than `quoteTimeoutMs` and ignores the late ones. If an estimate fails, its exception comes back through `get()` and is caught, so one bad estimate cannot crash the program. |
| 05 | Per shared object, why no data race is possible on it | section 5 below | See section 5: one row per shared object. |
| 05 | A type that is safe to copy or move | `Order.h` (copies deleted, moves `noexcept`); `Courier::View` (a plain copy) | `Order` cannot be copied, only moved, so a duplicate order does not even compile. `Courier::View` is a plain copy of a courier, so it can be read anywhere without a lock. |
| 06 | Mutual exclusion only through scope-bound locks; short critical sections | every `lock_guard` / `unique_lock`; `SlotGuard` in `shared/Restaurant.h` | Every lock is released automatically when its scope ends, even after an early `return` or an exception, and `SlotGuard` does the same for a pickup slot. Locks are held only while touching shared data, never while sleeping, so other threads wait very little. |
| 06 | One operation that needs two locks, and why it cannot deadlock | `Courier::transferAll()` with `std::scoped_lock` | When a courier breaks down, all its orders move to another courier at once, so nobody can see an order in both bags or in neither. `std::scoped_lock` takes both locks safely, so two breakdowns at the same moment cannot block each other; taking the locks one by one in opposite orders could deadlock. |
| 07 | Monitor objects | `EventLog`, `OrderBook`, `Restaurant`, `Courier`, `Fleet`, `EstimatePool` | Each shared object keeps its data private together with its mutex, and every method that touches data that can change takes the lock. Data that never changes after construction (a restaurant’s id, the list of couriers) is read without it. No thread can reach changing data without the lock, so nobody can forget it. |
| 07 | Threads that block until a condition holds, robust to empty wake-ups and to a final wake-up for shutdown | `OrderBook::take()`, `Courier::nextTask()` / `sleepUntil()`, `Restaurant::acquireSlot()`, `Fleet::waitForChange()`, `OrderGenerator::sleepUntil()`, `EstimatePool::workerLoop()`; every `stop()` / `close()` | Waiting threads sleep on a condition variable with a condition, so they use no CPU, and a wake-up with no news only makes them check again and go back to sleep. Every `stop()` or `close()` sets a flag and calls `notify_all()`, so all sleeping threads wake up at the end and the program stops in about a second. |

The project also uses folders 08 to 10 of the course. Folder 08: the flag set by Ctrl+C is a lock-free `std::atomic<bool>`, the only kind of shared variable that is safe to touch inside a signal handler, and the dispatcher's statistics are atomics with relaxed order because nobody uses them to order other data. Folder 09: the estimates run on a thread pool (example 55); experiment E1 shows why it is better than one `std::async` per estimate. Folder 10: at start-up, tasks take groups of nodes from an atomic counter to build the shortest-path tables in parallel, and every thread that makes random decisions has its own random generator, so there is no race and a seed always gives the same orders.


## 5. Why there is no data race

| Shared object | Shared by | Why no data race is possible |
|---|---|---|
| `Config`, `Settings` | every thread | TODO |
| `Graph` | couriers, dispatcher, estimate tasks, generator, window | TODO |
| `SimClock` | every thread | TODO |
| `EventLog` | generator, dispatcher, couriers, main | TODO |
| `OrderBook` | generator, dispatcher, window | TODO |
| `Restaurant` | couriers | TODO |
| `Courier` (bag, position, state) | its own thread, dispatcher, other couriers (handover), window | TODO |
| `Fleet` (version counter, courier list) | dispatcher, couriers | TODO |
| `EstimatePool` job queue | dispatcher, pool workers | TODO |
| `Courier::View` and the estimate inputs | estimate tasks, window | TODO |
| `g_stopRequested`, dispatcher statistics | signal handler, main, dispatcher | TODO |
| random generators | generator thread, each courier thread, dispatcher | TODO |
| rows of `dist_` / `next_` during start-up | shortest-path tasks | TODO |

ThreadSanitizer reported no race in any scenario (section 6).

## 6. Analysis with tools

Each tool observed a complete run of `config/equipetrol.json` with bursts of orders, breakdowns and a stop by signal. The unedited output of every run is in `analysis/`.

### ThreadSanitizer (races)

- **Version:** TODO (`g++ --version`)
- **Command:** TODO (copy the exact command)
- **Run checked:** TODO (number of orders, bursts and breakdowns in that log; the signal used)
- **What it reported:** TODO
- **What I changed:** TODO
- **Result of the re-run:** TODO
- **Output files:** `analysis/TODO`

### AddressSanitizer + LeakSanitizer (memory)

- **Version:** TODO
- **Command:** TODO
- **Run checked:** TODO
- **What it reported:** TODO
- **What I changed:** TODO
- **Result of the re-run:** TODO
- **Output files:** `analysis/TODO`

> TODO: list and explain every report that remains in the final version (for example, memory held by OpenCV, GTK or fontconfig until exit). An unexplained report counts as a defect.

## 7. Experiments

Every experiment changes one setting and keeps the configuration, the seed and the machine the same, with 3 runs per case. The scripts are `tools/run_experiments.sh` and `tools/summarize_runs.py`. The raw logs, summaries and stderr of every run, and the full table, are in `experiments/`.

### E1: how the estimates are computed (design change)

- **Changed:** `EQ_ESTIMATES` = `deferred` (one after another) / `async` (one new thread per estimate) / `pool`; then `async` and `pool` with one estimate per order slowed to 500 ms.
- **Kept equal:** `config/scenarios/fast.json`, seed 42, K = 8, `quoteTimeoutMs` = 200.
- **Measured:** average and maximum time to rank one order (`avgRankUs`, `maxRankUs`), late estimates, delivered orders.

| Case | avgRankUs | maxRankUs | late | delivered |
|---|---|---|---|---|
| deferred | TODO | TODO | TODO | TODO |
| async | TODO | TODO | TODO | TODO |
| pool | TODO | TODO | TODO | TODO |
| async, slow estimate | TODO | TODO | TODO | TODO |
| pool, slow estimate | TODO | TODO | TODO | TODO |

> TODO: conclusion. Explain why `deferred` is the fastest with tiny estimates but breaks the brief, why the pool is faster than `async`, and why with a slow estimate `async` overshoots `quoteTimeoutMs` while the pool stays at the deadline.

**Design change.** TODO: before = `async`, after = `pool`, with the two numbers that justify it.

### E2: hold the pickup slot while the food cooks, or wait for the food first

- **Changed:** `EQ_FOOD_WAIT` = `holding` / `before`.
- **Kept equal:** `config/scenarios/one_slot.json` (one slot per restaurant), seed 42.
- **Measured:** delivered orders, average and maximum delivery time.

| Case | delivered | avg delivery (min) | max delivery (min) |
|---|---|---|---|
| holding | TODO | TODO | TODO |
| before | TODO | TODO | TODO |

> TODO: conclusion. Be honest if the difference is inside the run-to-run noise.

### E3: number of candidates compared per order (K)

- **Changed:** `EQ_CANDIDATES` = 1, 2, 4, 8, 16.
- **Kept equal:** `config/scenarios/fast.json`, seed 42, pool mode.
- **Measured:** `avgRankUs`, delivered orders, average delivery time.

| K | avgRankUs | delivered | avg delivery (min) |
|---|---|---|---|
| 1 | TODO | TODO | TODO |
| 2 | TODO | TODO | TODO |
| 4 | TODO | TODO | TODO |
| 8 | TODO | TODO | TODO |
| 16 | TODO | TODO | TODO |

> TODO: conclusion.

### E4: parallel start-up of the shortest-path tables

- **Changed:** `EQ_GRAPH_TASKS` = 1, 2, 4, 8, 16 (dynamic chunks), and 8 tasks with static chunks (`EQ_GRAPH_GRAIN=0`).
- **Kept equal:** a 40 x 40 grid (1600 nodes, 3120 streets, `config/scenarios/grid40.json`), the machine (TODO: cores / `hardware_concurrency()`).
- **Measured:** time to build the tables (`graph:` line on stderr), speedup = T(1) / T(n), Amdahl's prediction 1 / (f + (1 − f) / n).

| Tasks | time (ms) | speedup | Amdahl |
|---|---|---|---|
| 1 | TODO | 1.0 | 1.0 |
| 2 | TODO | TODO | TODO |
| 4 | TODO | TODO | TODO |
| 8 | TODO | TODO | TODO |
| 16 | TODO | TODO | TODO |
| 8, static chunks | TODO | TODO | – |

> TODO: conclusion. How you estimated the serial fraction f, why the speedup flattens, and what static vs dynamic chunks show on a grid where every source costs the same.

## 8. Map and data

The background image and the street network come from OpenStreetMap (© OpenStreetMap contributors, ODbL). Details of the bounding box, the tools used to trace the network, and the restaurant positions are in `data/README.md`.
