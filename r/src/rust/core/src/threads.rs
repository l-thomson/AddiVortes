//! Independent work spread over a bounded number of scoped threads: the
//! chains of a fit, and the row chunks of a prediction.

/// The number of threads for `items` items: at most `threads`, at most
/// one per item, and at most the parallelism available to the process
/// (affinity masks and cgroup quotas included).
fn bound(threads: usize, items: usize) -> usize {
    let cores = std::thread::available_parallelism().map_or(usize::MAX, |c| c.get());
    threads.clamp(1, items.max(1)).min(cores)
}

/// Run `work` once on every item, the items split into at most `threads`
/// contiguous chunks, each chunk on a thread of its own; one thread runs
/// the items in order on the calling thread. A result that depends only
/// on its own item is the same on every thread count.
pub(crate) fn spread<T: Send>(items: &mut [T], threads: usize, work: impl Fn(&mut T) + Sync) {
    let threads = bound(threads, items.len());
    if threads == 1 {
        items.iter_mut().for_each(work);
        return;
    }
    let per = items.len().div_ceil(threads);
    let work = &work;
    std::thread::scope(|scope| {
        for chunk in items.chunks_mut(per) {
            scope.spawn(move || chunk.iter_mut().for_each(work));
        }
    });
}

/// Run `work` over `out` split into at most `threads` contiguous chunks,
/// each chunk on a thread of its own; `work(start, chunk)` receives the
/// chunk and the index of its first item. One thread runs the whole of
/// `out` on the calling thread.
pub(crate) fn spread_rows<T: Send>(
    out: &mut [T],
    threads: usize,
    work: impl Fn(usize, &mut [T]) + Sync,
) {
    let threads = bound(threads, out.len());
    if threads == 1 {
        work(0, out);
        return;
    }
    let per = out.len().div_ceil(threads);
    let work = &work;
    std::thread::scope(|scope| {
        for (index, chunk) in out.chunks_mut(per).enumerate() {
            scope.spawn(move || work(index * per, chunk));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{bound, spread, spread_rows};

    #[test]
    fn the_thread_count_is_bounded_by_the_items_and_the_machine() {
        let cores = std::thread::available_parallelism().map_or(1, |c| c.get());
        assert_eq!(bound(0, 10), 1);
        assert_eq!(bound(3, 0), 1);
        assert_eq!(bound(3, 2), 2.min(cores));
        assert_eq!(bound(1_000_000, 1_000_000), cores);
    }

    #[test]
    fn every_row_takes_its_own_index_on_any_thread_count() {
        for threads in [0, 1, 2, 3, 7, 40] {
            let mut rows: Vec<usize> = vec![0; 10];
            spread_rows(&mut rows, threads, |start, chunk| {
                for (offset, row) in chunk.iter_mut().enumerate() {
                    *row = start + offset;
                }
            });
            let expected: Vec<usize> = (0..10).collect();
            assert_eq!(rows, expected, "threads = {threads}");
        }
        let mut none: Vec<usize> = Vec::new();
        spread_rows(&mut none, 4, |_, chunk| assert!(chunk.is_empty()));
    }

    #[test]
    fn every_item_is_visited_once_on_any_thread_count() {
        for threads in [0, 1, 2, 3, 7, 40] {
            let mut items: Vec<usize> = vec![0; 10];
            spread(&mut items, threads, |v| *v += 1);
            assert!(items.iter().all(|&v| v == 1), "threads = {threads}");
        }
        let mut none: Vec<usize> = Vec::new();
        spread(&mut none, 4, |v| *v += 1);
        assert!(none.is_empty());
    }
}
