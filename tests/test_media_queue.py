"""Unit tests for multi-file / multi-image queue ingestion."""
import tempfile
import unittest
from pathlib import Path

from state_manager import StateManager


class TestMediaQueue(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test_queue.sqlite3"
        self.sm = StateManager(self.db_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_enqueue_and_fifo_order(self):
        chat_id = 1001
        rnc = "101068744"

        self.assertEqual(self.sm.get_queue_count(chat_id, rnc=rnc), 0)

        # Enqueue 3 items (mixed photos and pdfs)
        c1 = self.sm.enqueue_media(chat_id, "file_photo_1", "photo", rnc=rnc)
        self.assertEqual(c1, 1)
        c2 = self.sm.enqueue_media(chat_id, "file_pdf_2", "pdf", file_name="factura_combustible.pdf", rnc=rnc)
        self.assertEqual(c2, 2)
        c3 = self.sm.enqueue_media(chat_id, "file_photo_3", "photo", rnc=rnc)
        self.assertEqual(c3, 3)

        self.assertEqual(self.sm.get_queue_count(chat_id, rnc=rnc), 3)

        # Pop 1 (should be photo 1)
        item1 = self.sm.pop_next_queued_media(chat_id, rnc=rnc)
        self.assertIsNotNone(item1)
        self.assertEqual(item1["file_id"], "file_photo_1")
        self.assertEqual(item1["media_type"], "photo")
        self.assertEqual(self.sm.get_queue_count(chat_id, rnc=rnc), 2)

        # Pop 2 (should be pdf 2)
        item2 = self.sm.pop_next_queued_media(chat_id, rnc=rnc)
        self.assertIsNotNone(item2)
        self.assertEqual(item2["file_id"], "file_pdf_2")
        self.assertEqual(item2["file_name"], "factura_combustible.pdf")
        self.assertEqual(self.sm.get_queue_count(chat_id, rnc=rnc), 1)

        # Pop 3 (should be photo 3)
        item3 = self.sm.pop_next_queued_media(chat_id, rnc=rnc)
        self.assertIsNotNone(item3)
        self.assertEqual(item3["file_id"], "file_photo_3")
        self.assertEqual(self.sm.get_queue_count(chat_id, rnc=rnc), 0)

        # Pop on empty
        self.assertIsNone(self.sm.pop_next_queued_media(chat_id, rnc=rnc))

    def test_queue_isolation_between_chats_and_companies(self):
        # Chat 1, Empresa A
        self.sm.enqueue_media(1, "file_c1_a", "photo", rnc="101068744")
        # Chat 1, Empresa B
        self.sm.enqueue_media(1, "file_c1_b", "photo", rnc="101008172")
        # Chat 2, Empresa A
        self.sm.enqueue_media(2, "file_c2_a", "photo", rnc="101068744")

        self.assertEqual(self.sm.get_queue_count(1, rnc="101068744"), 1)
        self.assertEqual(self.sm.get_queue_count(1, rnc="101008172"), 1)
        self.assertEqual(self.sm.get_queue_count(2, rnc="101068744"), 1)

        item = self.sm.pop_next_queued_media(1, rnc="101068744")
        self.assertEqual(item["file_id"], "file_c1_a")
        self.assertEqual(self.sm.get_queue_count(1, rnc="101068744"), 0)
        # Otros siguen intactos
        self.assertEqual(self.sm.get_queue_count(1, rnc="101008172"), 1)
        self.assertEqual(self.sm.get_queue_count(2, rnc="101068744"), 1)

    def test_clear_media_queue(self):
        self.sm.enqueue_media(10, "f1", "photo")
        self.sm.enqueue_media(10, "f2", "pdf")
        self.assertEqual(self.sm.get_queue_count(10), 2)

        cleared = self.sm.clear_media_queue(10)
        self.assertEqual(cleared, 2)
        self.assertEqual(self.sm.get_queue_count(10), 0)


if __name__ == "__main__":
    unittest.main()
