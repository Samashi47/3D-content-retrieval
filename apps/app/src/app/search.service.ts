import { Injectable } from '@angular/core';
import {
  HttpClient,
  HttpErrorResponse,
  HttpHeaders,
} from '@angular/common/http';
import { API_URL } from './env';
import { map, shareReplay } from 'rxjs/operators';
import { Observable } from 'rxjs/internal/Observable';

interface searchResult {
  model_name: string;
  category: string;
  thumbnail: string;
  similarity: number;
}

interface descriptors {
  zernike: number[];
  fourier: number[];
}

@Injectable({
  providedIn: 'root',
})
export class SearchService {
  constructor(private http: HttpClient) {}

  downloadModel(filename: string, category: string): any {
    const headers = new HttpHeaders()
      .set('Accept', 'model/obj')
      .set('Content-Type', 'application/json');

    return this.http.post(
      `${API_URL}/download-model`,
      { filename, category },
      {
        headers,
        responseType: 'blob',
      }
    );
  }

  queryDescriptors(modelFile: Blob): Observable<descriptors> {
    const formData = new FormData();
    formData.append('model', modelFile);
    return this.http
      .post<descriptors>(`${API_URL}/query-descriptors`, formData)
      .pipe(shareReplay());
  }

  resultDescriptors(
    model_name: string,
    category: string
  ): Observable<descriptors> {
    return this.http
      .post<descriptors>(`${API_URL}/result-descriptors`, {
        model_name,
        category,
      })
      .pipe(shareReplay());
  }

  search(modelFile: Blob, numberOfResults: number): Observable<searchResult[]> {
    const formData = new FormData();
    formData.append('model', modelFile);
    formData.append('numberOfResults', numberOfResults.toString());
    console.log(formData);
    return this.http
      .post<searchResult[]>(`${API_URL}/search`, formData)
      .pipe(shareReplay());
  }
}
