# Author: [Denys Pylypenko] ([PI-41 mp])
# Date: 20 April 2025
# Description: This script implements video encryption and decryption using Python and OpenCV.
#              It processes video frames, encrypts them using a permutation algorithm, and decrypts them back.

import cv2
import numpy as np
from time import time, sleep

# Функція LFSR для генерації псевдовипадкової послідовності
def lfsr(seed, polynomial, length):
    state = seed
    sequence = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        feedback = 0
        for tap in polynomial:
            feedback ^= (state >> tap) & 1
        sequence[i] = state & 1
        state = (state >> 1) | (feedback << int(np.log2(seed)))
    return sequence

# Функція для генерації таблиці перестановок
def generate_permutation(seed, size):
    polynomial = [3, 2, 1, 0]
    seq = lfsr(seed, polynomial, 100)
    np.random.seed(int(sum(seq[:32])))
    perm = np.random.permutation(size)
    return perm

# Функція для сегментації зображення
def segment_image(img, rows, cols):
    h, w, _ = img.shape
    block_h = h // rows
    block_w = w // cols
    h_remainder = h % rows
    w_remainder = w % cols
    segments = np.zeros((rows, cols, block_h, block_w, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            curr_block_h = block_h + (1 if i == rows - 1 and h_remainder > 0 else 0)
            curr_block_w = block_w + (1 if j == cols - 1 and w_remainder > 0 else 0)
            if curr_block_h > 0 and curr_block_w > 0:
                segment = img[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w, :]
                if segment.shape[0] != block_h or segment.shape[1] != block_w:
                    segment = cv2.resize(segment, (block_w, block_h))
                segments[i, j] = segment
    return segments, block_h, block_w, img

# Функція для перестановки сегментів
def scramble_segments(segments, perm):
    rows, cols, block_h, block_w, _ = segments.shape
    total_blocks = rows * cols
    scrambled_segments = np.zeros_like(segments)
    segments_flat = segments.reshape(total_blocks, block_h, block_w, 3)
    idx = 0
    for i in range(total_blocks):
        r, c = np.unravel_index(perm[i], (rows, cols))
        scrambled_segments[r, c] = segments_flat[idx]
        idx += 1
    return scrambled_segments

# Функція для відновлення зображення
def reconstruct_image(segments, block_h, block_w, orig_h, orig_w):
    rows, cols, _, _, _ = segments.shape
    img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            y_start = i * block_h
            y_end = (i + 1) * block_h
            x_start = j * block_w
            x_end = (j + 1) * block_w
            if y_end > orig_h:
                y_end = orig_h
            if x_end > orig_w:
                x_end = orig_w
            target_h = y_end - y_start
            target_w = x_end - x_start
            if target_h > 0 and target_w > 0:
                segment = segments[i, j]
                if segment.shape[0] != target_h or segment.shape[1] != target_w:
                    segment = cv2.resize(segment, (target_w, target_h))
                img[y_start:y_end, x_start:x_end, :] = segment
    return img

# Функція шифрування
def encrypt_image(img, seed, rows, cols):
    h, w, _ = img.shape
    segments, block_h, block_w, resized_img = segment_image(img, rows, cols)
    perm = generate_permutation(seed, rows * cols)
    scrambled_segments = scramble_segments(segments, perm)
    encrypted_img = reconstruct_image(scrambled_segments, block_h, block_w, h, w)
    return encrypted_img, perm

# Функція дешифрування
def decrypt_image(encrypted_img, seed, rows, cols, perm):
    h, w, _ = encrypted_img.shape
    segments, block_h, block_w, resized_img = segment_image(encrypted_img, rows, cols)
    inv_perm = np.zeros_like(perm)
    for i in range(len(perm)):
        inv_perm[perm[i]] = i
    scrambled_segments = scramble_segments(segments, inv_perm)
    decrypted_img = reconstruct_image(scrambled_segments, block_h, block_w, h, w)
    return decrypted_img

# Функція для обчислення схожості
def calculate_similarity(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_flat = img1_gray.flatten().astype(np.float32)
    img2_flat = img2_gray.flatten().astype(np.float32)
    img1_flat -= np.mean(img1_flat)
    img2_flat -= np.mean(img2_flat)
    numerator = np.sum(img1_flat * img2_flat)
    denominator = np.sqrt(np.sum(img1_flat**2) * np.sum(img2_flat**2))
    if denominator == 0:
        return 0.0
    corr = numerator / denominator
    similarity = corr * 100
    return similarity

# Основний скрипт для шифрування, дешифрування та відображення відео
def main():
    # Параметри
    seed = 44257
    rows = 40
    cols = 30
    frame_rate = 30
    frame_duration = 1 / frame_rate
    similarity_interval = 30  # Обчислювати схожість кожні 30 кадрів
    display_interval = 2  # Відображати кадри кожні 2 кадри

    # Завантаження відеофайлу
    try:
        vid = cv2.VideoCapture('test_video.mp4')
    except Exception as e:
        print(f"Не вдалося завантажити відеофайл: {e}")
        return

    # Отримання інформації про відео
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Обробка відео...")

    # Основний цикл для шифрування, дешифрування та відображення
    frame_idx = 1
    total_time = 0
    total_latency = 0
    num_processed_frames = 0
    similarity_orig_enc_list = []
    similarity_orig_dec_list = []

    # Створюємо вікно для відображення
    cv2.namedWindow('Video Playback', cv2.WINDOW_NORMAL)

    while frame_idx <= num_frames:
        start_time = time()

        # Читання кадру
        ret, img = vid.read()
        if not ret:
            break

        # Перевірка коректності кадру
        if img is None or len(img.shape) != 3 or img.shape[2] != 3:
            print(f"Пропущено кадр {frame_idx}: Некоректний формат кадру.")
            frame_idx += 1
            continue

        # Зменшення розміру кадру для прискорення обробки
        try:
            target_h, target_w = 480, 640  # (height, width)
            img = cv2.resize(img, (target_w, target_h))  # cv2.resize очікує (width, height)
        except Exception as e:
            print(f"Помилка при зміні розміру кадру {frame_idx}: {e}")
            frame_idx += 1
            continue

        # Початок вимірювання затримки
        tic_latency = time()

        # Шифрування
        encrypted_img, perm = encrypt_image(img, seed, rows, cols)

        # Дешифрування
        decrypted_img = decrypt_image(encrypted_img, seed, rows, cols, perm)

        # Кінець вимірювання затримки
        latency = time() - tic_latency
        total_latency += latency
        num_processed_frames += 1

        # Обчислення схожості в реальному часі (кожні similarity_interval кадрів)
        if frame_idx % similarity_interval == 0:
            h_enc, w_enc, _ = encrypted_img.shape
            img_resized = cv2.resize(img, (w_enc, h_enc))
            similarity_orig_enc = calculate_similarity(img_resized, encrypted_img)
            similarity_orig_dec = calculate_similarity(img_resized, decrypted_img)
            similarity_orig_enc_list.append(similarity_orig_enc)
            similarity_orig_dec_list.append(similarity_orig_dec)
            # print(f"Кадр {frame_idx}: Similarity (Original vs Encrypted): {similarity_orig_enc:.2f}%")
            # print(f"Кадр {frame_idx}: Similarity (Original vs Decrypted): {similarity_orig_dec:.2f}%")

        # Відображення кадрів (кожні display_interval кадрів)
        if frame_idx % display_interval == 0:
            # Додаємо заголовки до зображень
            h, w, _ = img.shape
            img_with_title = np.zeros((h + 30, w, 3), dtype=np.uint8)
            img_with_title[30:, :, :] = img
            cv2.putText(img_with_title, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            encrypted_with_title = np.zeros((h + 30, w, 3), dtype=np.uint8)
            encrypted_with_title[30:, :, :] = encrypted_img
            cv2.putText(encrypted_with_title, "Encrypted", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            decrypted_with_title = np.zeros((h + 30, w, 3), dtype=np.uint8)
            decrypted_with_title[30:, :, :] = decrypted_img
            cv2.putText(decrypted_with_title, "Decrypted", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 250), 2)

            # Об’єднуємо зображення горизонтально
            combined_img = np.hstack((img_with_title, encrypted_with_title, decrypted_with_title))
            cv2.imshow('Video Playback', combined_img)

            # Перевірка натискання клавіші для завершення
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC для завершення
                print("Відтворення перервано користувачем.")
                break

            # Перевірка, чи вікно не закрите
            if cv2.getWindowProperty('Video Playback', cv2.WND_PROP_VISIBLE) < 1:
                print("Вікно закрите користувачем.")
                break

        # Вимірювання загального часу обробки кадру
        elapsed_time = time() - start_time
        total_time += elapsed_time

        # Контроль частоти кадрів
        if elapsed_time < frame_duration:
            sleep(frame_duration - elapsed_time)

        frame_idx += 1

    # Виведення результатів
    vid.release()
    cv2.destroyAllWindows()
    print(f"Загальний час обробки {frame_idx-1} кадрів: {total_time:.2f} секунд")
    if num_processed_frames > 0:
        avg_latency = total_latency / num_processed_frames
        print(f"Середня затримка між оригінальним і дешифрованим зображенням: {avg_latency:.4f} секунд")
    else:
        print("Не вдалося обчислити середню затримку: немає оброблених кадрів.")

    # Виведення середньої схожості
    if similarity_orig_enc_list:
        avg_similarity_orig_enc = np.mean(similarity_orig_enc_list)
        avg_similarity_orig_dec = np.mean(similarity_orig_dec_list)
        print(f"Середня схожість (Original vs Encrypted): {avg_similarity_orig_enc:.2f}%")
        print(f"Середня схожість (Original vs Decrypted): {avg_similarity_orig_dec:.2f}%")
    else:
        print("Не вдалося обчислити середню схожість: немає обчислених значень.")

    print("Обробка завершена.")

if __name__ == "__main__":
    main()