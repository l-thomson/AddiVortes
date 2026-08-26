//! Independent work spread over a bounded number of scoped threads: the
//! chains of a fit, and the row chunks of a prediction.

/// Run `work` once on every item, the items split into at most `threads`
/// contiguous chunks, each chunk on a thread of its own; one thread runs
/// the items in order on the calling thread. A result that depends only
/// on its own item is the same on every thread count.
pub(crate) fn spread<T: Send>(items: &mut [T], threads: usize, work: impl Fn(&mut T) + Sync) {
    let threads = threads.clamp(1, items.len().max(1));
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

#[cfg(test)]
mod tests {
    use super::spread;

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
